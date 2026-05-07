# Install script for directory: /DREAMPlace/dreamplace/ops/dct

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/DREAMPlace/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so"
         RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct" TYPE MODULE FILES "/DREAMPlace/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so"
         OLD_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib:"
         NEW_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_cpp.cpython-38-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so"
         RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct" TYPE MODULE FILES "/DREAMPlace/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so"
         OLD_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib:"
         NEW_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct_lee_cpp.cpython-38-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so"
         RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct" TYPE MODULE FILES "/DREAMPlace/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so"
         OLD_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib:"
         NEW_RPATH "/opt/conda/lib/python3.8/site-packages/torch/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct/dct2_fft2_cpp.cpython-38-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/dreamplace/ops/dct" TYPE FILE FILES
    "/DREAMPlace/dreamplace/ops/dct/__init__.py"
    "/DREAMPlace/dreamplace/ops/dct/dct.py"
    "/DREAMPlace/dreamplace/ops/dct/dct2_fft2.py"
    "/DREAMPlace/dreamplace/ops/dct/dct_lee.py"
    "/DREAMPlace/dreamplace/ops/dct/discrete_spectral_transform.py"
    "/DREAMPlace/dreamplace/ops/dct/naive.py"
    "/DREAMPlace/dreamplace/ops/dct/torch_fft_api.py"
    )
endif()

