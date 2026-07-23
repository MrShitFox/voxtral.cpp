if(NOT DEFINED GGML_SOURCE_DIR)
    message(FATAL_ERROR "GGML_SOURCE_DIR is required")
endif()

set(vulkan_source "${GGML_SOURCE_DIR}/src/ggml-vulkan/ggml-vulkan.cpp")
file(READ "${vulkan_source}" source_text)

set(old_filter
"        case GGML_OP_CONCAT:
            return ggml_type_size(op->src[0]->type) == ggml_type_size(GGML_TYPE_F32);")
set(new_filter
"        case GGML_OP_CONCAT:
            // The dispatch table below already provides both concat_f32 and
            // concat_f16.  Admit FP16 here so the scheduler does not route an
            // otherwise device-resident KV window through the CPU.
            return op->src[0]->type == op->src[1]->type &&
                   op->src[0]->type == op->type &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);")

string(FIND "${source_text}" "${new_filter}" already_patched)
if(NOT already_patched EQUAL -1)
    message(STATUS "GGML Vulkan FP16 CONCAT capability patch already applied")
    return()
endif()

string(FIND "${source_text}" "${old_filter}" old_filter_offset)
if(old_filter_offset EQUAL -1)
    message(FATAL_ERROR
        "Pinned GGML Vulkan CONCAT capability filter changed; refusing an unverified patch")
endif()

string(REPLACE "${old_filter}" "${new_filter}" source_text "${source_text}")
file(WRITE "${vulkan_source}" "${source_text}")
message(STATUS "Applied GGML Vulkan FP16 CONCAT capability patch")
