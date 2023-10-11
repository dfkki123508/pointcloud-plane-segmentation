cmake_minimum_required(VERSION 3.10)
project(o3dapp)

include(FetchContent)

set(CMAKE_CXX_STANDARD 14)

find_package(Open3D)
find_package(OpenMP)
find_package(Eigen3 CONFIG REQUIRED)

# Open3D
if (NOT Open3D_FOUND)
    
  set(DEP_NAME "Open3D")
  string(TOLOWER ${DEP_NAME} DEP_NAME_LOWER)

  message(STATUS "Downloading ${DEP_NAME} to ${FETCHCONTENT_BASE_DIR}/${DEP_NAME_LOWER}-src")

  FetchContent_Declare(${DEP_NAME}
    URL https://github.com/isl-org/Open3D/releases/download/v0.17.0/open3d-devel-windows-amd64-0.17.0.zip
    URL_HASH SHA256=4262ff8e9bf8ba0b453b9593d0cdf6f0bbaac54b384277cd0bfe94653f33b5c8
    TLS_VERIFY true
    DOWNLOAD_EXTRACT_TIMESTAMP true
  )

  FetchContent_MakeAvailable(${DEP_NAME})

  set(Open3D_DIR "${FETCHCONTENT_BASE_DIR}/${DEP_NAME_LOWER}-src/CMake")

  find_package(Open3D CONFIG REQUIRED)

endif()

add_subdirectory(rspd)
get_target_property(LIBB_INCLUDES rspd INCLUDE_DIRECTORIES)
message(STATUS "rspd includes: ${LIBB_INCLUDES}")

add_executable(main src/main.cpp)
# Directories to the linker

target_link_libraries(main PRIVATE Open3D::Open3D rspd Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
endif()


# copy dlls
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${rspd_BINARY_DIR}/$<CONFIGURATION>/"
        $<TARGET_FILE_DIR:main>)

add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/Users/user/dev/pointcloud-plane-segmentation/build/_deps/open3d-src/bin/"
            $<TARGET_FILE_DIR:main>)