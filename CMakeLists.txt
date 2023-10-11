cmake_minimum_required(VERSION 3.10)
project(o3dapp)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
find_package(Eigen3 CONFIG REQUIRED)

find_package(PCL 1.12 REQUIRED COMPONENTS io octree visualization registration)
message(STATUS "PCL components at ${PCL_COMPONENTS}")
message(STATUS "PCL headers at ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL libraries: ${PCL_LIBRARIES}")
message(STATUS "PCL definitions: ${PCL_DEFINITIONS}")

add_subdirectory(rspd)
get_target_property(LIBB_INCLUDES rspd INCLUDE_DIRECTORIES)
message(STATUS "rspd includes: ${LIBB_INCLUDES}")

add_executable(main src/main.cpp)

target_include_directories(main PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(main PRIVATE VTK_LEGACY_SILENT -DNO_EXPLICIT_INSTANTIATIONS)
target_compile_options(main PRIVATE ${PCL_COMPILE_OPTIONS})
target_link_libraries(main PRIVATE ${PCL_LIBRARIES} rspd Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()


# copy dlls
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${rspd_BINARY_DIR}/$<CONFIGURATION>/"
        $<TARGET_FILE_DIR:main>)
