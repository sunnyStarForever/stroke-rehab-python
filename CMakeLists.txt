cmake_minimum_required(VERSION 3.16)
project(rehab_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# STROKE_ENGINE_STUB=ON  → Windows dev: Config+Logger only, no hardware libs
# STROKE_ENGINE_STUB=OFF → Linux AArch64: full engine with OpenCV/OpenNI2/ONNX
# ============================================================
option(STROKE_ENGINE_STUB
  "Build stub engine for desktop dev validation (excludes camera/pose/EMG)"
  OFF)

# ---- pybind11 ----
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_pybind11_cmake_dir)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
  endif()
  find_package(pybind11 REQUIRED)
endif()

# ---- Python ----
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python: ${Python_EXECUTABLE} (${Python_VERSION})")

if(STROKE_ENGINE_STUB)
  # ==============================
  # STUB MODE — zero external deps
  # ==============================
  message(STATUS "=== STUB BUILD (desktop dev validation) ===")

  # Only compile binding code + Logger callback mechanism.
  # Config types are header-only and exposed directly from the binding.
  set(ENGINE_STUB_SOURCES
    bindings/module.cpp
    engine/util/FpsCounter.cpp
  )

  pybind11_add_module(_core ${ENGINE_STUB_SOURCES})
  target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(_core PRIVATE STROKE_ENGINE_STUB=1)
  if(MSVC)
    target_compile_options(_core PRIVATE /utf-8)
  endif()

else()
  # ==============================
  # FULL MODE — Linux AArch64 target
  # ==============================
  message(STATUS "=== FULL BUILD (Linux AArch64 target) ===")

  # ---- OpenCV ----
  set(OpenCV_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../opencv-4.4.0/build
    ${CMAKE_CURRENT_SOURCE_DIR}/../opencv/build
    /usr/lib/aarch64-linux-gnu/cmake/opencv4
    /usr/local/lib/cmake/opencv4
  )
  if(NOT OpenCV_DIR)
    foreach(_hint IN LISTS OpenCV_HINTS)
      if(EXISTS "${_hint}/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_hint}" CACHE PATH "OpenCV cmake dir" FORCE)
        break()
      endif()
    endforeach()
  endif()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  message(STATUS "OpenCV: ${OpenCV_VERSION}")

  # ---- OpenNI2 ----
  find_path(OpenNI2_INCLUDE_DIR OpenNI.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../including/OpenNI/sdk/Include
          /usr/include/openni2  /usr/include/ni2
          /usr/local/include/openni2  /usr/local/include/ni2)
  find_library(OpenNI2_LIBRARY NAMES OpenNI2
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../including/OpenNI/sdk/libs
          /usr/lib  /usr/local/lib)
  if(OpenNI2_INCLUDE_DIR AND OpenNI2_LIBRARY)
    set(HAVE_OPENNI2_DEF TRUE)
  else()
    set(HAVE_OPENNI2_DEF FALSE)
  endif()

  # ---- ONNX Runtime ----
  set(ONNXRUNTIME_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../including/onnxruntime
    /usr  /usr/local  /opt/onnxruntime)
  if(ONNXRUNTIME_DIR)
    list(PREPEND ONNXRUNTIME_HINTS ${ONNXRUNTIME_DIR})
  endif()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_HINTS}  PATH_SUFFIXES include include/onnxruntime)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS ${ONNXRUNTIME_HINTS}  PATH_SUFFIXES lib lib64)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(HAVE_ONNXRUNTIME_DEF TRUE)
  else()
    set(HAVE_ONNXRUNTIME_DEF FALSE)
  endif()

  # ---- Engine sources ----
  set(ENGINE_SOURCES
    engine/capture/RgbCaptureV4L2.cpp
    engine/capture/DepthCaptureOpenNI.cpp
    engine/sync/SyncManager.cpp
    engine/sync/TimestampNormalizer.cpp
    engine/align/HardwareD2CAligner.cpp
    engine/align/SoftwareRegistrationAligner.cpp
    engine/pose/PoseEstimator.cpp
    engine/pose/PoseEstimatorRTMPoseOrt.cpp
    engine/pose/PersonDetectorOrt.cpp
    engine/pose/AdaptiveRoiBoundingBoxProvider.cpp
    engine/pose/DepthSampler.cpp
    engine/pose/JointProjector3D.cpp
    engine/pose/EMASkeletonFilter.cpp
    engine/pose/SkeletonSmoother.cpp
    engine/pose/Halpe26ToRehab22Mapper.cpp
    engine/pose/JointNameMapper.cpp
    engine/emg/EmgRpmsgClient.cpp
    engine/util/FpsCounter.cpp
    engine/util/ThreadAffinity.cpp
  )

  # ---- Static lib ----
  add_library(rehab_engine_core STATIC ${ENGINE_SOURCES})
  target_include_directories(rehab_engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}  ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(rehab_engine_core PUBLIC ${OpenCV_LIBS})

  if(HAVE_OPENNI2_DEF)
    target_include_directories(rehab_engine_core PUBLIC ${OpenNI2_INCLUDE_DIR})
    target_link_libraries(rehab_engine_core PUBLIC ${OpenNI2_LIBRARY})
    target_compile_definitions(rehab_engine_core PUBLIC HAVE_OPENNI2=1)
  endif()
  if(HAVE_ONNXRUNTIME_DEF)
    target_include_directories(rehab_engine_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(rehab_engine_core PUBLIC ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(rehab_engine_core PUBLIC HAVE_ONNXRUNTIME=1)
  endif()

  target_compile_definitions(rehab_engine_core PRIVATE
    DEFAULT_CALIBRATION_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../configs/calibration.yaml\"
    DEFAULT_POSE_MODEL_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../including/rtmpose-t/end2end.onnx\"
    DEFAULT_POSE_DETECTOR_MODEL_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../including/yolov8n/yolov8n.onnx\"
    DEFAULT_POSE_PIPELINE_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../including/rtmpose-t/pipeline.json\"
    DEFAULT_POSE_DETAIL_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../including/rtmpose-t/detail.json\"
    DEFAULT_POSE_DEPLOY_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/../including/rtmpose-t/deploy.json\"
  )

  if(UNIX)
    target_link_libraries(rehab_engine_core PUBLIC pthread)
  endif()

  # GCC < 10 ICE workaround
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
    message(STATUS "Applying GCC < 10 ICE workaround (compiling heavy files with -O0)")
    foreach(_src ${ENGINE_SOURCES})
      if(_src MATCHES "AdaptiveRoiBoundingBoxProvider" OR
         _src MATCHES "DepthSampler" OR
         _src MATCHES "PoseEstimatorRTMPoseOrt" OR
         _src MATCHES "PersonDetectorOrt" OR
         _src MATCHES "SoftwareRegistrationAligner" OR
         _src MATCHES "SyncManager" OR
         _src MATCHES "RgbCaptureV4L2")
        set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "-O0")
        message(STATUS "  -O0: ${_src}")
      endif()
    endforeach()
  endif()

  # ---- pybind11 module ----
  pybind11_add_module(_core bindings/module.cpp)
  target_link_libraries(_core PRIVATE rehab_engine_core)

  message(STATUS "  HAVE_OPENNI2:      ${HAVE_OPENNI2_DEF}")
  message(STATUS "  HAVE_ONNXRUNTIME:  ${HAVE_ONNXRUNTIME_DEF}")
endif()

# ---- Common output settings ----
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

message(STATUS "  Output: ${CMAKE_BINARY_DIR}/_core${CMAKE_SHARED_MODULE_SUFFIX}")