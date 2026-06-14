cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Configurable paths
#
# QWEN3_TTS_SOURCE_DIR  : path to the qwen3-tts.cpp checkout (contains ggml/
#                         as a submodule and the project CMakeLists.txt).
# QWEN3_TTS_BINARY_DIR  : where to place qwen3-tts build artifacts inside the
#                         server build tree (optional, cosmetic).
# ---------------------------------------------------------------------------
set(QWEN3_TTS_SOURCE_DIR "" CACHE PATH
    "Path to the qwen3-tts.cpp source directory (with ggml submodule)")

if(NOT QWEN3_TTS_SOURCE_DIR)
    # Default: sibling location inside the edge-tts repo
    set(QWEN3_TTS_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../research/qwen3-tts/qwen3-tts.cpp")
    get_filename_component(QWEN3_TTS_SOURCE_DIR "${QWEN3_TTS_SOURCE_DIR}" ABSOLUTE)
endif()

if(NOT EXISTS "${QWEN3_TTS_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "QWEN3_TTS_SOURCE_DIR (${QWEN3_TTS_SOURCE_DIR}) does not contain "
        "a CMakeLists.txt. Set -DQWEN3_TTS_SOURCE_DIR=/path/to/qwen3-tts.cpp")
endif()

message(STATUS "qwen3-tts source: ${QWEN3_TTS_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Pull in the full qwen3-tts.cpp build tree.
#
# This gives us the `qwen3_tts` static library target (C++ API — Qwen3TTS
# class, save_audio_file, load_audio_file) and all transitive GGML
# dependencies, ready to link against.
#
# We also build `qwen3tts_shared` as a side effect — the C API shared lib —
# but the server links the static `qwen3_tts` target for direct C++ access.
#
# GGML must already be built under ${QWEN3_TTS_SOURCE_DIR}/ggml/build before
# configuring this project (the upstream CMakeLists references those paths).
# ---------------------------------------------------------------------------
set(QWEN3_TTS_BINARY_DIR "${CMAKE_BINARY_DIR}/qwen3-tts-build" CACHE PATH
    "Binary directory for the nested qwen3-tts build")

add_subdirectory(
    "${QWEN3_TTS_SOURCE_DIR}"
    "${QWEN3_TTS_BINARY_DIR}"
    EXCLUDE_FROM_ALL
)

# ---------------------------------------------------------------------------
# Server executable
# ---------------------------------------------------------------------------
add_executable(qwen3-tts-server
    server.cpp
)

target_include_directories(qwen3-tts-server PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"          # httplib.h, json.hpp (local)
)

# Link against the static qwen3_tts library (full C++ API).
# GGML + Threads come transitively through the qwen3_tts target's
# PUBLIC link dependencies.
target_link_libraries(qwen3-tts-server PRIVATE
    qwen3_tts
)

# Suppress warnings from the vendored headers
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen3-tts-server PRIVATE -Wno-unused-parameter)
endif()
