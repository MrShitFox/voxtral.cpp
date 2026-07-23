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

# Standard testing switch. include(CTest) defines the cache option BUILD_TESTING
# (default ON) and calls enable_testing(). The internal C++ tests and the
# streaming-skeleton harness below are gated on it, so a production build with
# -DBUILD_TESTING=OFF builds only the installed CLI tools. The existing
# local/remote build+ctest workflow keeps working with no extra flags.
include(CTest)

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
        GIT_TAG        5cecdad692d868e28dbd2f7c468504770108f30c
        # This pinned revision already ships an FP16 Vulkan CONCAT pipeline,
        # but its supports_op() filter only admits four-byte elements.  Encoder
        # FP16 KV therefore falls back to the CPU for every layer.  Apply the
        # narrow, locally tested capability fix while retaining the otherwise-
        # proven dependency revision.
        PATCH_COMMAND  ${CMAKE_COMMAND}
                       -DGGML_SOURCE_DIR=<SOURCE_DIR>
                       -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch-ggml-vulkan-f16-concat.cmake)
    FetchContent_MakeAvailable(ggml)
    set(VOXTRAL_GGML_TARGETS ggml)
    set(VOXTRAL_GGML_INCLUDES "")
endif()

add_library(voxtral_lib
    src/voxtral.cpp
    src/voxtral-mel.cpp
    src/voxtral-stream.cpp)

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

# ----------------------------------------------------------------------------
# Tests (standard BUILD_TESTING). enable_testing() was already called by
# include(CTest) above. A production build (-DBUILD_TESTING=OFF) skips this
# block entirely, so no test-only executable is created. The internal streaming
# header lives in src/.
# ----------------------------------------------------------------------------
if(BUILD_TESTING)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME voxtral_python_reference
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_voxtral_reference.py)
    endif()

    # Model-free unit tests for the stream lifecycle / state machine / PCM path.
    add_executable(voxtral_stream_unit tests/cpp/test_stream_state.cpp)
    target_link_libraries(voxtral_stream_unit PRIVATE voxtral_lib)
    target_include_directories(voxtral_stream_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_stream_unit COMMAND voxtral_stream_unit)

    # Model-free numerical parity + memory-bound tests for the incremental Mel
    # frontend (batch vs incremental, one-shot frames, bounded PCM retention).
    add_executable(voxtral_mel_unit tests/cpp/test_mel_incremental.cpp)
    target_link_libraries(voxtral_mel_unit PRIVATE voxtral_lib)
    target_include_directories(voxtral_mel_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_mel_unit COMMAND voxtral_mel_unit)

    # Model-free schedule/arithmetic tests for the incremental causal encoder
    # (batch-vs-incremental chunk schedule equivalence, conv frame math, bounded
    # work). End-to-end tensor parity is proven by the Node.js GPU acceptance suite.
    add_executable(voxtral_encoder_unit tests/cpp/test_encoder_incremental.cpp)
    target_link_libraries(voxtral_encoder_unit PRIVATE voxtral_lib)
    target_include_directories(voxtral_encoder_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_encoder_unit COMMAND voxtral_encoder_unit)

    # Model-free ring-planner tests for the per-layer encoder KV-cache (append,
    # wrap, eviction, causal-window mask, absolute positions across rollover, conv
    # window alignment). End-to-end tensor parity is proven by the GPU suite.
    add_executable(voxtral_encoder_kv_unit tests/cpp/test_encoder_kv.cpp)
    target_link_libraries(voxtral_encoder_kv_unit PRIVATE voxtral_lib)
    target_include_directories(voxtral_encoder_kv_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_encoder_kv_unit COMMAND voxtral_encoder_kv_unit)

    # Model-free decoder KV ring arithmetic: absolute positions, modulo writes,
    # logical read ordering, wrap and fully-sliding eviction without movement.
    add_executable(voxtral_decoder_kv_unit tests/cpp/test_decoder_kv.cpp)
    target_link_libraries(voxtral_decoder_kv_unit PRIVATE voxtral_lib)
    target_include_directories(voxtral_decoder_kv_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_decoder_kv_unit COMMAND voxtral_decoder_kv_unit)

    # Model-driven harness used by the Node.js acceptance suite. Not a CLI/server.
    add_executable(voxtral-stream-test tests/cpp/voxtral_stream_test.cpp)
    target_link_libraries(voxtral-stream-test PRIVATE voxtral_lib)
    target_include_directories(voxtral-stream-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

#add_subdirectory(examples/stream)
