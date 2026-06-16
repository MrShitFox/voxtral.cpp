cmake_minimum_required(VERSION 3.16)

project(voxtral_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VOXTRAL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(VOXTRAL_NATIVE_OPT "Enable native CPU tuning flags" ON)
# Homebrew uses the system ggml formula; everywhere else ggml is built from source
# via FetchContent (no vendored submodule). Backends are discovered at runtime via
# the ggml registry, so no backend flags are needed here.
option(VOXTRAL_USE_SYSTEM_GGML "Link a system ggml instead of building it from source" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

if(VOXTRAL_USE_SYSTEM_GGML)
    find_package(ggml CONFIG REQUIRED)
    set(VOXTRAL_GGML_TARGETS ggml::ggml ggml::ggml-base)
    # Imported targets omit the include dir with dynamically-loaded backends.
    set(VOXTRAL_GGML_INCLUDES ${GGML_INCLUDE_DIR})
else()
    include(FetchContent)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(APPLE)
        # Embed Metal shaders so the binary needs no external default.metallib.
        set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG        5cecdad692d868e28dbd2f7c468504770108f30c)
    FetchContent_MakeAvailable(ggml)
    set(VOXTRAL_GGML_TARGETS ggml)
    set(VOXTRAL_GGML_INCLUDES "")
endif()

add_library(voxtral_lib
    src/voxtral.cpp)

# On Windows with BUILD_SHARED_LIBS=ON, export symbols automatically so the
# import library (voxtral_lib.lib) is generated for dependents like `voxtral`.
if(WIN32 AND BUILD_SHARED_LIBS)
    set_target_properties(voxtral_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_include_directories(voxtral_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${VOXTRAL_GGML_INCLUDES})

target_link_libraries(voxtral_lib
    PUBLIC
        ${VOXTRAL_GGML_TARGETS}
        Threads::Threads)

if(NOT WIN32)
    target_link_libraries(voxtral_lib PUBLIC m)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(voxtral_lib PUBLIC ${CMAKE_DL_LIBS})
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral_lib PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    if(APPLE)
        message(STATUS "VOXTRAL_NATIVE_OPT: skipping -march/-mtune=native on Apple toolchains")
    else()
        target_compile_options(voxtral_lib PRIVATE -march=native -mtune=native)
    endif()
endif()

add_executable(voxtral src/main.cpp)
target_link_libraries(voxtral PRIVATE voxtral_lib)

add_executable(voxtral-quantize src/voxtral-quantize.cpp)
target_link_libraries(voxtral-quantize PRIVATE ${VOXTRAL_GGML_TARGETS} Threads::Threads)
target_include_directories(voxtral-quantize PRIVATE ${VOXTRAL_GGML_INCLUDES})

if(NOT WIN32)
    target_link_libraries(voxtral-quantize PRIVATE m)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(voxtral-quantize PRIVATE ${CMAKE_DL_LIBS})
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral-quantize PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral-quantize PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    if(NOT APPLE)
        target_compile_options(voxtral-quantize PRIVATE -march=native -mtune=native)
    endif()
endif()

# Install the CLI tools (used by `cmake --install` and the Homebrew formula).
install(TARGETS voxtral voxtral-quantize RUNTIME DESTINATION bin)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_custom_target(voxtral_convert
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/convert_voxtral_to_gguf.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print converter usage"
        VERBATIM)
    add_custom_target(voxtral_quantize_all
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/quantize_all.py --help
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Print quantize_all usage"
        VERBATIM)
endif()

enable_testing()
if(Python3_Interpreter_FOUND)
    add_test(
        NAME voxtral_python_reference
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_voxtral_reference.py)
endif()

#add_subdirectory(examples/stream)
