cmake_minimum_required(VERSION 3.16)

project(voxtral_cpp LANGUAGES C CXX)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VOXTRAL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(VOXTRAL_NATIVE_OPT "Enable native CPU tuning flags" ON)
option(VOXTRAL_BUILD_EXAMPLES "Build public API examples" ON)
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
# -DBUILD_TESTING=OFF skips every test-only executable. The existing
# local/remote build+ctest workflow keeps working with no extra flags.
include(CTest)

find_package(Threads REQUIRED)

# The standalone network adapter is enabled by default when its normal system
# dependencies are discoverable. Boost.System is header-only on the supported
# Boost range. Some distributions (including Arch) ship libboost_json without
# per-component CMake config files, so discover the headers and JSON library
# separately rather than requiring boost_systemConfig.cmake.
find_package(Boost 1.75 QUIET)
find_library(VOXTRAL_BOOST_JSON_LIBRARY
    NAMES boost_json
    HINTS ${Boost_LIBRARY_DIRS})
if(Boost_FOUND AND VOXTRAL_BOOST_JSON_LIBRARY)
    set(VOXTRAL_BUILD_SERVER_DEFAULT ON)
else()
    set(VOXTRAL_BUILD_SERVER_DEFAULT OFF)
endif()
option(
    VOXTRAL_BUILD_SERVER
    "Build the standalone HTTP/WebSocket transcription server"
    ${VOXTRAL_BUILD_SERVER_DEFAULT})
if(VOXTRAL_BUILD_SERVER AND
   (NOT Boost_FOUND OR NOT VOXTRAL_BOOST_JSON_LIBRARY))
    message(FATAL_ERROR
        "VOXTRAL_BUILD_SERVER requires Boost >= 1.75 headers and libboost_json")
endif()
if(VOXTRAL_BUILD_SERVER)
    if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR
            "voxtral-server must link libvoxtral.so.1; "
            "do not configure it with BUILD_SHARED_LIBS=OFF")
    endif()
    set(BUILD_SHARED_LIBS ON CACHE BOOL
        "Build shared libraries (required by voxtral-server)")
endif()

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

set(VOXTRAL_CORE_SOURCES
    src/voxtral.cpp
    src/voxtral-mel.cpp
    src/voxtral-stream.cpp
    src/voxtral-stream-diagnostics.cpp
    src/voxtral-stream-events.cpp
    src/voxtral-stream-telemetry.cpp
    src/voxtral-stream-frontend.cpp
    src/voxtral-stream-decoder.cpp)

# Compile the inference runtime once with hidden visibility. The CLI and
# internal acceptance tests consume a private static archive, while the public
# library combines the same PIC objects with the explicitly exported C adapter.
# This keeps every C++/test helper out of the shared-library ABI.
add_library(voxtral_core OBJECT ${VOXTRAL_CORE_SOURCES})
set_target_properties(voxtral_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(voxtral_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${VOXTRAL_GGML_INCLUDES})
target_link_libraries(voxtral_core
    PRIVATE
        ${VOXTRAL_GGML_TARGETS}
        Threads::Threads)

add_library(voxtral_internal STATIC $<TARGET_OBJECTS:voxtral_core>)
set_target_properties(voxtral_internal PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(voxtral_internal
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${VOXTRAL_GGML_INCLUDES})

add_library(voxtral_lib
    src/voxtral-c-api.cpp
    $<TARGET_OBJECTS:voxtral_core>)
set_target_properties(voxtral_lib PROPERTIES
    OUTPUT_NAME voxtral
    VERSION 1.0.0
    SOVERSION 1
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(UNIX)
    if(APPLE)
        set_property(TARGET voxtral_lib PROPERTY
            INSTALL_RPATH "@loader_path;@loader_path/../lib64")
    else()
        set_property(TARGET voxtral_lib PROPERTY
            INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib64")
    endif()
endif()
target_compile_definitions(voxtral_lib PRIVATE
    $<$<BOOL:${BUILD_SHARED_LIBS}>:VOXTRAL_BUILD_SHARED>)
target_include_directories(voxtral_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${VOXTRAL_GGML_INCLUDES})
if(BUILD_SHARED_LIBS AND UNIX)
    if(APPLE)
        target_link_options(voxtral_lib PRIVATE
            "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/voxtral.exports")
    else()
        target_link_options(voxtral_lib PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/voxtral.map")
    endif()
endif()

foreach(voxtral_target IN ITEMS voxtral_internal voxtral_lib)
    target_link_libraries(${voxtral_target}
        PUBLIC
            ${VOXTRAL_GGML_TARGETS}
            Threads::Threads)
endforeach()

if(NOT WIN32)
    target_link_libraries(voxtral_internal PUBLIC m)
    target_link_libraries(voxtral_lib PUBLIC m)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(voxtral_internal PUBLIC ${CMAKE_DL_LIBS})
    target_link_libraries(voxtral_lib PUBLIC ${CMAKE_DL_LIBS})
endif()

if(VOXTRAL_WARNINGS_AS_ERRORS)
    if (MSVC)
        target_compile_options(voxtral_core PRIVATE /W4 /WX)
        target_compile_options(voxtral_lib PRIVATE /W4 /WX)
    else()
        target_compile_options(voxtral_core PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(voxtral_lib PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endif()

if(VOXTRAL_NATIVE_OPT AND NOT MSVC)
    if(APPLE)
        message(STATUS "VOXTRAL_NATIVE_OPT: skipping -march/-mtune=native on Apple toolchains")
    else()
        target_compile_options(voxtral_core PRIVATE -march=native -mtune=native)
        target_compile_options(voxtral_lib PRIVATE -march=native -mtune=native)
    endif()
endif()

add_executable(voxtral src/main.cpp)
target_link_libraries(voxtral PRIVATE voxtral_internal)

if(VOXTRAL_BUILD_EXAMPLES)
    add_executable(voxtral-streaming-c examples/c/streaming.c)
    set_target_properties(voxtral-streaming-c PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        LINKER_LANGUAGE CXX)
    target_link_libraries(voxtral-streaming-c PRIVATE voxtral_lib)
endif()

if(VOXTRAL_BUILD_SERVER)
    add_library(voxtral_server_common STATIC
        server/auth.cpp
        server/gpu-lease.cpp
        server/json-protocol.cpp
        server/logger.cpp
        server/server-config.cpp
        server/wav-reader.cpp)
    target_include_directories(voxtral_server_common
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/server
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${Boost_INCLUDE_DIRS})
    target_link_libraries(voxtral_server_common
        PUBLIC
            ${VOXTRAL_BOOST_JSON_LIBRARY}
            Threads::Threads)
    target_compile_definitions(voxtral_server_common PUBLIC
        BOOST_ERROR_CODE_HEADER_ONLY)

    add_executable(voxtral-server
        server/batch-transcription.cpp
        server/engine.cpp
        server/http-session.cpp
        server/main.cpp
        server/realtime-session.cpp
        server/server-app.cpp)
    target_include_directories(voxtral-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/server)
    target_link_libraries(voxtral-server PRIVATE
        voxtral_server_common
        voxtral_lib
        Threads::Threads)
    if(UNIX)
        if(APPLE)
            set_property(TARGET voxtral-server PROPERTY
                INSTALL_RPATH "@loader_path/../lib")
        else()
            set_property(TARGET voxtral-server PROPERTY
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN/../lib")
        endif()
    endif()
endif()

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

# Install the CLI tools, stable library and C headers.
install(TARGETS voxtral voxtral-quantize
    RUNTIME DESTINATION bin)
if(VOXTRAL_BUILD_SERVER)
    install(TARGETS voxtral-server RUNTIME DESTINATION bin)
endif()
install(TARGETS voxtral_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
install(FILES
    include/voxtral.h
    include/voxtral-stream.h
    DESTINATION include)

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
    if(VOXTRAL_BUILD_SERVER)
        add_executable(voxtral_server_unit
            tests/server/server_unit_test.cpp)
        target_link_libraries(voxtral_server_unit PRIVATE
            voxtral_server_common)
        add_test(NAME voxtral_server_unit COMMAND voxtral_server_unit)
    endif()

    # Public headers are compiled as standalone translation units without any
    # private include directory or GGML header dependency.
    add_executable(public_header_c11 tests/headers/public_header_smoke.c)
    set_target_properties(public_header_c11 PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF)
    target_include_directories(public_header_c11 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME public_header_c11 COMMAND public_header_c11)

    add_executable(public_header_c17 tests/headers/public_header_smoke.c)
    set_target_properties(public_header_c17 PROPERTIES
        C_STANDARD 17
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF)
    target_include_directories(public_header_c17 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME public_header_c17 COMMAND public_header_c17)

    add_executable(public_header_cpp17 tests/headers/public_header_smoke.cpp)
    set_target_properties(public_header_cpp17 PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
    target_include_directories(public_header_cpp17 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME public_header_cpp17 COMMAND public_header_cpp17)

    add_executable(public_header_cpp20 tests/headers/public_header_smoke.cpp)
    set_target_properties(public_header_cpp20 PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
    target_include_directories(public_header_cpp20 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME public_header_cpp20 COMMAND public_header_cpp20)

    # Model-free C ABI contract tests. The source is compiled as C and links
    # only to the documented public library.
    add_executable(voxtral_public_api_unit tests/c/test_public_api.c)
    set_target_properties(voxtral_public_api_unit PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
        LINKER_LANGUAGE CXX)
    target_link_libraries(voxtral_public_api_unit PRIVATE voxtral_lib)
    add_test(NAME voxtral_public_api_unit COMMAND voxtral_public_api_unit)

    # Model-driven public API harness. It is a C translation unit and links only
    # against the installed/public library target; no src/ include directory or
    # private helper symbol is available.
    add_executable(voxtral_public_stream_api_test
        tests/c/public_stream_api_test.c)
    set_target_properties(voxtral_public_stream_api_test PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS ON
        LINKER_LANGUAGE CXX)
    target_link_libraries(voxtral_public_stream_api_test PRIVATE voxtral_lib)

    # CPU-only microbenchmark of the ABI validation wrapper itself. Comparing
    # two zero-length calls on one stream excludes GPU work and pacing sleeps.
    add_executable(voxtral_public_api_call_benchmark
        tests/cpp/benchmark_public_api_call.cpp
        src/voxtral-c-api.cpp)
    target_link_libraries(voxtral_public_api_call_benchmark
        PRIVATE voxtral_internal)

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME voxtral_python_reference
            COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_voxtral_reference.py)
    endif()

    # Model-free unit tests for the stream lifecycle / state machine / PCM path.
    add_executable(voxtral_stream_unit tests/cpp/test_stream_state.cpp)
    target_link_libraries(voxtral_stream_unit PRIVATE voxtral_internal)
    target_include_directories(voxtral_stream_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_stream_unit COMMAND voxtral_stream_unit)

    # Model-free numerical parity + memory-bound tests for the incremental Mel
    # frontend (batch vs incremental, one-shot frames, bounded PCM retention).
    add_executable(voxtral_mel_unit tests/cpp/test_mel_incremental.cpp)
    target_link_libraries(voxtral_mel_unit PRIVATE voxtral_internal)
    target_include_directories(voxtral_mel_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_mel_unit COMMAND voxtral_mel_unit)

    # Model-free schedule/arithmetic tests for the incremental causal encoder
    # (batch-vs-incremental chunk schedule equivalence, conv frame math, bounded
    # work). End-to-end tensor parity is proven by the Node.js GPU acceptance suite.
    add_executable(voxtral_encoder_unit tests/cpp/test_encoder_incremental.cpp)
    target_link_libraries(voxtral_encoder_unit PRIVATE voxtral_internal)
    target_include_directories(voxtral_encoder_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_encoder_unit COMMAND voxtral_encoder_unit)

    # Model-free ring-planner tests for the per-layer encoder KV-cache (append,
    # wrap, eviction, causal-window mask, absolute positions across rollover, conv
    # window alignment). End-to-end tensor parity is proven by the GPU suite.
    add_executable(voxtral_encoder_kv_unit tests/cpp/test_encoder_kv.cpp)
    target_link_libraries(voxtral_encoder_kv_unit PRIVATE voxtral_internal)
    target_include_directories(voxtral_encoder_kv_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_encoder_kv_unit COMMAND voxtral_encoder_kv_unit)

    # Model-free decoder KV ring arithmetic: absolute positions, modulo writes,
    # logical read ordering, wrap and fully-sliding eviction without movement.
    add_executable(voxtral_decoder_kv_unit tests/cpp/test_decoder_kv.cpp)
    target_link_libraries(voxtral_decoder_kv_unit PRIVATE voxtral_internal)
    target_include_directories(voxtral_decoder_kv_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_decoder_kv_unit COMMAND voxtral_decoder_kv_unit)

    # Header-only unit test for the streaming diagnostic SHA-256 (known-answer
    # vectors + chunk-invariance). Self-contained: no backend / model link.
    add_executable(voxtral_stream_diag_unit tests/cpp/test_stream_diagnostics.cpp)
    target_include_directories(voxtral_stream_diag_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME voxtral_stream_diag_unit COMMAND voxtral_stream_diag_unit)

    # Model-driven harness used by the Node.js acceptance suite. Not a CLI/server.
    add_executable(voxtral-stream-test tests/cpp/voxtral_stream_test.cpp)
    target_link_libraries(voxtral-stream-test PRIVATE voxtral_internal)
    target_include_directories(voxtral-stream-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

#add_subdirectory(examples/stream)
