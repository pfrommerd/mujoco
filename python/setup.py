# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Install script for MuJoCo."""

import fnmatch
import logging
import os
import platform
import random
import re
import shutil
import string
import subprocess
import sys
import sysconfig

import setuptools
from setuptools.command import build_ext
from setuptools.command import install_scripts

MUJOCO_CMAKE = 'MUJOCO_CMAKE'
MUJOCO_CMAKE_ARGS = 'MUJOCO_CMAKE_ARGS'
MUJOCO_PATH = 'MUJOCO_PATH'
MUJOCO_PLUGIN_PATH = 'MUJOCO_PLUGIN_PATH'
_BUNDLED_MUJOCO_SOURCE_DIR = os.path.join(
    os.path.dirname(__file__), 'mujoco', '_src'
)
_BUNDLED_MUJOCO_SOURCE_EXISTS = os.path.exists(
    os.path.join(_BUNDLED_MUJOCO_SOURCE_DIR, 'CMakeLists.txt')
)
_BUILD_SIMULATE_EXTENSION = (
    MUJOCO_PATH in os.environ or _BUNDLED_MUJOCO_SOURCE_EXISTS
)
_PLUGIN_LIBRARY_NAME_HINTS = (
    'actuator',
    'elasticity',
    'sensor',
    'sdf_plugin',
    'usd_decoder_plugin',
)

EXT_PREFIX = 'mujoco.'


def get_long_description():
  """Creates a long description for the package from bundled markdown files."""
  current_dir = os.path.dirname('__file__')
  with open(os.path.join(current_dir, 'README.md')) as f:
    description = f.read()
  try:
    with open(os.path.join(current_dir, 'LICENSES_THIRD_PARTY.md')) as f:
      description = f'{description}\n{f.read()}'
  except FileNotFoundError:
    pass
  return description


def get_mujoco_lib_pattern():
  if platform.system() == 'Windows':
    return 'mujoco.lib'
  elif platform.system() == 'Darwin':
    return 'libmujoco.*.dylib'
  else:
    return 'libmujoco.so.*'


def get_external_lib_patterns():
  if platform.system() == 'Windows':
    return ['mujoco.dll']
  elif platform.system() == 'Darwin':
    return ['libmujoco.*.dylib']
  else:
    return ['libmujoco.so.*']


def get_plugin_lib_patterns():
  if platform.system() == 'Windows':
    return ['*.dll']
  elif platform.system() == 'Darwin':
    return ['lib*.dylib']
  else:
    return ['lib*']


def start_and_end(iterable):
  it = iter(iterable)
  while True:
    try:
      first = next(it)
      second = next(it)
      yield first, second
    except StopIteration:
      return


def tokenize_quoted_substr(input_string, quote_char, placeholders=None):
  """Replace quoted substrings with random text placeholders with no spaces."""
  # Matches quote characters not proceded with a backslash.
  pattern = re.compile(r'(?<!\\)' + quote_char)
  quote_positions = [m.start() for m in pattern.finditer(input_string)]
  if len(quote_positions) % 2:
    raise ValueError(f'unbalanced quotes {quote_char}...{quote_char}')

  output_string = ''
  placeholders = placeholders if placeholders is not None else dict()
  prev_end = -1
  for start, end in start_and_end(quote_positions):
    output_string += input_string[prev_end + 1 : start]
    while True:
      placeholder = ''.join(random.choices(string.ascii_lowercase, k=5))
      if placeholder not in input_string and placeholder not in output_string:
        break
    output_string += placeholder
    placeholders[placeholder] = input_string[start + 1 : end]
    prev_end = end
  output_string += input_string[prev_end + 1 :]

  return output_string, placeholders


def parse_cmake_args_from_environ(env_var_name=MUJOCO_CMAKE_ARGS):
  """Parses CMake arguments from an environment variable."""
  raw_args = os.environ.get(env_var_name, '').strip()
  unquoted, placeholders = tokenize_quoted_substr(raw_args, '"')
  unquoted, placeholders = tokenize_quoted_substr(unquoted, "'", placeholders)
  parts = re.split(r'\s+', unquoted.strip())
  out = []
  for part in parts:
    for k, v in placeholders.items():
      part = part.replace(k, v)
    part = part.replace('\\"', '"').replace("\\'", "'")
    if part:
      out.append(part)
  return out


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._is_apple = platform.system() == 'Darwin'
    self._mujoco_framework_path = None
    self._mujoco_library_path = None
    self._mujoco_include_path = None
    self._mujoco_plugins_path = None
    self._use_bundled_mujoco_source = False
    self._configure_mujoco_inputs()
    self._configure_cmake()
    self._remove_disabled_extensions_from_cmake_cache()
    for ext in self.extensions:
      assert ext.name.startswith(EXT_PREFIX)
      assert '.' not in ext.name[len(EXT_PREFIX) :]
      self.build_extension(ext)
    if self._use_bundled_mujoco_source:
      self._copy_bundled_mujoco_libraries()
      self._copy_bundled_mujoco_headers()
      self._copy_bundled_plugin_libraries()
    else:
      self._copy_external_libraries()
      self._copy_mujoco_headers()
      self._copy_plugin_libraries()
    if self._is_apple:
      self._copy_mjpython()

  def _remove_disabled_extensions_from_cmake_cache(self):
    build_simulate = self._get_cmake_cache_bool('MUJOCO_PYTHON_BUILD_SIMULATE')
    if build_simulate is False:
      self.extensions = [
          ext for ext in self.extensions if ext.name != 'mujoco._simulate'
      ]
      logging.warning(
          'Skipping mujoco._simulate extension because CMake disabled '
          'MUJOCO_PYTHON_BUILD_SIMULATE.'
      )

  def _get_cmake_cache_bool(self, key):
    cache_path = os.path.join(self.build_temp, 'CMakeCache.txt')
    if not os.path.exists(cache_path):
      return None
    with open(cache_path, encoding='utf-8') as f:
      for line in f:
        prefix = f'{key}:BOOL='
        if line.startswith(prefix):
          value = line[len(prefix) :].strip().upper()
          if value in ('ON', 'TRUE', '1', 'YES'):
            return True
          if value in ('OFF', 'FALSE', '0', 'NO'):
            return False
          return None
    return None

  def _configure_mujoco_inputs(self):
    if MUJOCO_PATH in os.environ:
      (
          self._mujoco_library_path,
          self._mujoco_include_path,
          self._mujoco_plugins_path,
          self._mujoco_framework_path,
      ) = self._find_mujoco()
      return

    bundled_cmake_file = os.path.join(_BUNDLED_MUJOCO_SOURCE_DIR, 'CMakeLists.txt')
    if os.path.exists(bundled_cmake_file):
      self._use_bundled_mujoco_source = True
      self._mujoco_include_path = os.path.join(_BUNDLED_MUJOCO_SOURCE_DIR, 'include')
      return

    raise RuntimeError(
        f'Neither {MUJOCO_PATH} environment variable nor bundled MuJoCo source'
        ' tree are available'
    )

  def _find_mujoco(self):
    if MUJOCO_PATH not in os.environ:
      raise RuntimeError(f'{MUJOCO_PATH} environment variable is not set')
    if MUJOCO_PLUGIN_PATH not in os.environ:
      raise RuntimeError(
          f'{MUJOCO_PLUGIN_PATH} environment variable is not set'
      )
    library_path = None
    include_path = None
    plugin_path = os.environ[MUJOCO_PLUGIN_PATH]
    for directory, subdirs, filenames in os.walk(os.environ[MUJOCO_PATH]):
      if self._is_apple and 'mujoco.framework' in subdirs:
        return (
            os.path.join(directory, 'mujoco.framework/Versions/A'),
            os.path.join(directory, 'mujoco.framework/Headers'),
            plugin_path,
            directory,
        )
      if fnmatch.filter(filenames, get_mujoco_lib_pattern()):
        library_path = directory
      if os.path.exists(os.path.join(directory, 'mujoco/mujoco.h')):
        include_path = directory
      if library_path and include_path:
        return library_path, include_path, plugin_path, None
    raise RuntimeError('Cannot find MuJoCo library and/or include paths')

  def _copy_external_libraries(self):
    dst = os.path.dirname(self.get_ext_fullpath(self.extensions[0].name))
    for directory, _, filenames in os.walk(os.environ[MUJOCO_PATH]):
      for pattern in get_external_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          shutil.copyfile(
              os.path.join(directory, filename), os.path.join(dst, filename)
          )

  def _copy_bundled_mujoco_libraries(self):
    dst = os.path.dirname(self.get_ext_fullpath(self.extensions[0].name))
    copied = []
    for directory, _, filenames in os.walk(self.build_temp):
      for pattern in get_external_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          src = os.path.join(directory, filename)
          out = os.path.join(dst, filename)
          shutil.copyfile(src, out)
          copied.append(out)
    if not copied:
      raise RuntimeError(
          f'Cannot find built MuJoCo library files in {self.build_temp}'
      )

  def _copy_bundled_plugin_libraries(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'plugin',
    )
    copied = 0
    os.makedirs(dst, exist_ok=True)
    for directory, _, filenames in os.walk(self.build_temp):
      for filename in filenames:
        lowercase_name = filename.lower()
        if not any(hint in lowercase_name for hint in _PLUGIN_LIBRARY_NAME_HINTS):
          continue
        if platform.system() == 'Windows':
          if not lowercase_name.endswith('.dll'):
            continue
        elif platform.system() == 'Darwin':
          if not lowercase_name.endswith('.dylib'):
            continue
        elif '.so' not in lowercase_name:
          continue
        shutil.copyfile(os.path.join(directory, filename), os.path.join(dst, filename))
        copied += 1
    if not copied:
      logging.warning(
          'No bundled plugin shared libraries were found in build output at %s',
          self.build_temp,
      )

  def _copy_bundled_mujoco_headers(self):
    self._copy_mujoco_headers()

  def _copy_plugin_libraries(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'plugin',
    )
    os.makedirs(dst)
    for directory, _, filenames in os.walk(self._mujoco_plugins_path):
      for pattern in get_plugin_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          shutil.copyfile(
              os.path.join(directory, filename), os.path.join(dst, filename)
          )

  def _copy_mujoco_headers(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'include/mujoco',
    )
    os.makedirs(dst)
    for directory, _, filenames in os.walk(self._mujoco_include_path):
      for filename in fnmatch.filter(filenames, '*.h'):
        shutil.copyfile(
            os.path.join(directory, filename), os.path.join(dst, filename)
        )

  def _copy_mjpython(self):
    src_dir = os.path.join(os.path.dirname(__file__), 'mujoco/mjpython')
    dst_contents_dir = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'MuJoCo_(mjpython).app/Contents',
    )
    os.makedirs(dst_contents_dir)
    shutil.copyfile(
        os.path.join(src_dir, 'Info.plist'),
        os.path.join(dst_contents_dir, 'Info.plist'),
    )

    dst_bin_dir = os.path.join(dst_contents_dir, 'MacOS')
    os.makedirs(dst_bin_dir)
    shutil.copyfile(
        os.path.join(self.build_temp, 'mjpython'),
        os.path.join(dst_bin_dir, 'mjpython'),
    )
    os.chmod(os.path.join(dst_bin_dir, 'mjpython'), 0o755)

    dst_resources_dir = os.path.join(dst_contents_dir, 'Resources')
    os.makedirs(dst_resources_dir)
    shutil.copyfile(
        os.path.join(src_dir, 'mjpython.icns'),
        os.path.join(dst_resources_dir, 'mjpython.icns'),
    )

  def _configure_cmake(self):
    """Check for CMake."""
    cmake = os.environ.get(MUJOCO_CMAKE, 'cmake')
    build_cfg = 'Debug' if self.debug else 'Release'
    cmake_module_path = os.path.join(
        os.path.dirname(__file__), 'mujoco', 'cmake'
    )
    cmake_args = [
        f'-DPython3_ROOT_DIR:PATH={sys.prefix}',
        f'-DPython3_EXECUTABLE:STRING={sys.executable}',
        f'-DCMAKE_MODULE_PATH:PATH={cmake_module_path}',
        f'-DCMAKE_BUILD_TYPE:STRING={build_cfg}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH={self.build_temp}',
        (
            f'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL={"OFF" if self.debug else "ON"}'
        ),
        '-DCMAKE_Fortran_COMPILER:STRING=',
        '-DBUILD_TESTING:BOOL=OFF',
        (
            f'-DMUJOCO_PYTHON_BUILD_SIMULATE:BOOL='
            f'{"ON" if _BUILD_SIMULATE_EXTENSION else "OFF"}'
        ),
    ]

    if self._mujoco_framework_path is not None:
      cmake_args.extend([
          f'-DMUJOCO_FRAMEWORK_DIR:PATH={self._mujoco_framework_path}',
      ])
    elif self._use_bundled_mujoco_source:
      cmake_args.extend([
          f'-DMUJOCO_PYTHON_MUJOCO_SOURCE_DIR:PATH={_BUNDLED_MUJOCO_SOURCE_DIR}',
      ])
    else:
      cmake_args.extend([
          f'-DMUJOCO_LIBRARY_DIR:PATH={self._mujoco_library_path}',
          f'-DMUJOCO_INCLUDE_DIR:PATH={self._mujoco_include_path}',
      ])

    if platform.system() != 'Windows':
      cmake_args.extend([
          f'-DPython3_LIBRARY={sysconfig.get_paths()["stdlib"]}',
          f'-DPython3_INCLUDE_DIR={sysconfig.get_paths()["include"]}',
      ])
    if platform.system() == 'Darwin' and os.environ.get('ARCHFLAGS'):
      osx_archs = []
      if '-arch x86_64' in os.environ['ARCHFLAGS']:
        osx_archs.append('x86_64')
      if '-arch arm64' in os.environ['ARCHFLAGS']:
        osx_archs.append('arm64')
      cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={";".join(osx_archs)}')

    cmake_args.extend(parse_cmake_args_from_environ())
    os.makedirs(self.build_temp, exist_ok=True)

    if platform.system() == 'Windows':
      cmake_args = [arg.replace('\\', '/') for arg in cmake_args]

    print('Configuring CMake with the following arguments:')
    for arg in cmake_args:
      print(f'    {arg}')
    subprocess.check_call(
        [cmake]
        + cmake_args
        + [os.path.join(os.path.dirname(__file__), 'mujoco')],
        cwd=self.build_temp,
    )

    print('Building all extensions with CMake')
    subprocess.check_call(
        [cmake, '--build', '.', f'-j{os.cpu_count()}', '--config', build_cfg],
        cwd=self.build_temp,
    )

  def build_extension(self, ext):
    dest_path = self.get_ext_fullpath(ext.name)
    build_path = os.path.join(self.build_temp, os.path.basename(dest_path))
    shutil.copyfile(build_path, dest_path)


class InstallScripts(install_scripts.install_scripts):
  """Strips file extension from executable scripts whose names end in `.py`."""

  def run(self):
    super().run()
    oldfiles = self.outfiles
    files = set(oldfiles)
    self.outfiles = []
    for oldfile in oldfiles:
      if oldfile.endswith('.py'):
        newfile = oldfile[:-3]
      else:
        newfile = oldfile

      renamed = False
      if newfile not in files and not os.path.exists(newfile):
        if not self.dry_run:
          os.rename(oldfile, newfile)
        renamed = True

      if renamed:
        logging.info(
            'Renaming %s script to %s',
            os.path.basename(oldfile),
            os.path.basename(newfile),
        )
        self.outfiles.append(newfile)
        files.remove(oldfile)
        files.add(newfile)
      else:
        self.outfiles.append(oldfile)


setuptools.setup(
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    cmdclass=dict(
        build_ext=BuildCMakeExtension,
        install_scripts=InstallScripts,
    ),
    ext_modules=[
        CMakeExtension('mujoco._callbacks'),
        CMakeExtension('mujoco._constants'),
        CMakeExtension('mujoco._enums'),
        CMakeExtension('mujoco._errors'),
        CMakeExtension('mujoco._functions'),
        CMakeExtension('mujoco._render'),
        CMakeExtension('mujoco._rollout'),
        CMakeExtension('mujoco._specs'),
        CMakeExtension('mujoco._structs'),
    ]
    + ([CMakeExtension('mujoco._simulate')] if _BUILD_SIMULATE_EXTENSION else []),
    scripts=['mujoco/mjpython/mjpython.py']
    if platform.system() == 'Darwin'
    else [],
)
