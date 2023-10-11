cmake_minimum_required(VERSION 3.10)
project(rspd)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

add_library(rspd SHARED src/planedetector.cpp src/planarpatch.cpp)
target_include_directories(rspd PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(rspd PRIVATE VTK_LEGACY_SILENT -DNO_EXPLICIT_INSTANTIATIONS)
target_compile_options(rspd PRIVATE ${PCL_COMPILE_OPTIONS})
target_link_libraries(rspd PRIVATE ${PCL_LIBRARIES})
target_include_directories(rspd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
