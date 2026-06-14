#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <ggml.h>
#include <gguf.h>
}

namespace {

constexpr const char * kVoxtralArch = "voxtral_realtime";

struct quant_options {
    std::string name;
    ggml_type base_type = GGML_TYPE_COUNT;
    uint32_t file_type = 0;
    bool q4_k_m = false;
};

struct tensor_ref {
    std::string name;
    ggml_tensor * tensor = nullptr;
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

void write_zeros(std::ofstream & out, size_t n) {
    static const std::array<char, 4096> zeros = {};
    while (n > 0) {
        const size_t chunk = std::min(n, zeros.size());
        out.write(zeros.data(), static_cast<std::streamsize>(chunk));
        n -= chunk;
    }
}

std::string tensor_shape_str(const ggml_tensor * t) {
    std::string out = "[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0) {
            break;
        }
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(t->ne[i]);
    }
    out += "]";
    return out;
}

std::string get_kv_str(const gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        return "";
    }
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        return "";
    }
    const char * val = gguf_get_val_str(gguf, id);
    return val ? std::string(val) : std::string();
}

quant_options parse_quant_type(const std::string & raw) {
    const std::string q = to_lower(raw);

    if (q == "q4_0") {
        return {"Q4_0", GGML_TYPE_Q4_0, GGML_FTYPE_MOSTLY_Q4_0, false};
    }
    if (q == "q4_1") {
        return {"Q4_1", GGML_TYPE_Q4_1, GGML_FTYPE_MOSTLY_Q4_1, false};
    }
    if (q == "q5_0") {
        return {"Q5_0", GGML_TYPE_Q5_0, GGML_FTYPE_MOSTLY_Q5_0, false};
    }
    if (q == "q5_1") {
        return {"Q5_1", GGML_TYPE_Q5_1, GGML_FTYPE_MOSTLY_Q5_1, false};
    }
    if (q == "q8_0" || q == "q8") {
        return {"Q8_0", GGML_TYPE_Q8_0, GGML_FTYPE_MOSTLY_Q8_0, false};
    }
    if (q == "q2_k") {
        return {"Q2_K", GGML_TYPE_Q2_K, GGML_FTYPE_MOSTLY_Q2_K, false};
    }
    if (q == "q3_k") {
        return {"Q3_K", GGML_TYPE_Q3_K, GGML_FTYPE_MOSTLY_Q3_K, false};
    }
    if (q == "q4_k") {
        return {"Q4_K", GGML_TYPE_Q4_K, GGML_FTYPE_MOSTLY_Q4_K, false};
    }
    if (q == "q5_k") {
        return {"Q5_K", GGML_TYPE_Q5_K, GGML_FTYPE_MOSTLY_Q5_K, false};
    }
    if (q == "q6_k" || q == "q6") {
        return {"Q6_K", GGML_TYPE_Q6_K, GGML_FTYPE_MOSTLY_Q6_K, false};
    }
    if (q == "q4_k_m") {
        return {"Q4_K_M", GGML_TYPE_Q4_K, GGML_FTYPE_MOSTLY_Q4_K, true};
    }

    throw std::runtime_error("unsupported quantization type '" + raw + "'");
}

const std::vector<std::regex> & voxtral_quantizable_patterns() {
    static const std::vector<std::regex> patterns = {
        std::regex(R"(^tok_embeddings\.weight$)"),
        std::regex(R"(^adapter\.(0|2)\.weight$)"),
        std::regex(R"(^enc\.blk\.[0-9]+\.(attn_(q|k|v|o)|ffn_w[123])\.weight$)"),
        std::regex(R"(^dec\.blk\.[0-9]+\.(attn_(q|k|v|o)|ffn_w[123]|ada[02])\.weight$)"),
        std::regex(R"(^output\.weight$)"),
        std::regex(R"(^output_mm\.weight$)"),
    };
    return patterns;
}

bool is_voxtral_quantizable(const std::string & name, const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    if (ggml_n_dims(t) != 2) {
        return false;
    }
    for (const auto & pattern : voxtral_quantizable_patterns()) {
        if (std::regex_match(name, pattern)) {
            return true;
        }
    }
    return false;
}

bool q4_k_m_prefers_q6_k(const std::string & name) {
    if (name == "tok_embeddings.weight" || name == "output.weight" || name == "output_mm.weight") {
        return true;
    }
    return name.find(".attn_v.weight") != std::string::npos || name.find(".ffn_w2.weight") != std::string::npos;
}

ggml_type choose_tensor_type(const quant_options & opts, const std::string & name) {
    if (!opts.q4_k_m) {
        return opts.base_type;
    }
    return q4_k_m_prefers_q6_k(name) ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K;
}

bool is_row_compatible(ggml_type type, int64_t n_per_row) {
    const int64_t qk = ggml_blck_size(type);
    return qk > 0 && (n_per_row % qk == 0);
}

size_t tensor_nbytes_for_type(const ggml_tensor * tensor, ggml_type type) {
    const int64_t ne0 = tensor->ne[0];
    if (ne0 <= 0) {
        return 0;
    }
    const int64_t nrows = ggml_nelements(tensor) / ne0;
    return ggml_row_size(type, ne0) * static_cast<size_t>(nrows);
}

void convert_rows_to_f32(const ggml_tensor * tensor, int64_t row0, int64_t nrows, float * dst) {
    if (tensor == nullptr || dst == nullptr) {
        throw std::runtime_error("invalid tensor conversion input");
    }

    const int64_t n_per_row = tensor->ne[0];
    const size_t src_row_size = ggml_row_size(tensor->type, n_per_row);
    const uint8_t * src_base = static_cast<const uint8_t *>(tensor->data) + static_cast<size_t>(row0) * src_row_size;

    if (tensor->type == GGML_TYPE_F32 && src_row_size == static_cast<size_t>(n_per_row) * sizeof(float)) {
        std::memcpy(dst, src_base, static_cast<size_t>(nrows) * src_row_size);
        return;
    }

    if (tensor->type == GGML_TYPE_F16) {
        for (int64_t r = 0; r < nrows; ++r) {
            const auto * src = reinterpret_cast<const ggml_fp16_t *>(src_base + static_cast<size_t>(r) * src_row_size);
            ggml_fp16_to_fp32_row(src, dst + r * n_per_row, n_per_row);
        }
        return;
    }

#ifdef GGML_TYPE_BF16
    if (tensor->type == GGML_TYPE_BF16) {
        for (int64_t r = 0; r < nrows; ++r) {
            const auto * src = reinterpret_cast<const ggml_bf16_t *>(src_base + static_cast<size_t>(r) * src_row_size);
            ggml_bf16_to_fp32_row(src, dst + r * n_per_row, n_per_row);
        }
        return;
    }
#endif

    if (ggml_is_quantized(tensor->type)) {
        const auto * traits = ggml_get_type_traits(tensor->type);
        if (traits->to_float == nullptr) {
            throw std::runtime_error(
                "cannot dequantize tensor type " + std::string(ggml_type_name(tensor->type)) + " (missing to_float)");
        }
        for (int64_t r = 0; r < nrows; ++r) {
            const void * src = src_base + static_cast<size_t>(r) * src_row_size;
            traits->to_float(src, dst + r * n_per_row, n_per_row);
        }
        return;
    }

    throw std::runtime_error("unsupported source tensor type " + std::string(ggml_type_name(tensor->type)));
}

size_t quantize_tensor_rows(
    const ggml_tensor * tensor,
    ggml_type dst_type,
    int nthread,
    std::ofstream & fout) {
    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = ggml_nelements(tensor) / n_per_row;
    const size_t dst_row_size = ggml_row_size(dst_type, n_per_row);

    // About 4 MiB of f32 conversion buffer per chunk.
    constexpr int64_t kChunkElems = 1LL << 20;
    int64_t chunk_rows = std::max<int64_t>(1, kChunkElems / std::max<int64_t>(1, n_per_row));
    chunk_rows = std::min<int64_t>(chunk_rows, nrows);

    std::vector<uint8_t> q_chunk(static_cast<size_t>(chunk_rows) * dst_row_size);
    std::vector<float> f32_chunk(static_cast<size_t>(chunk_rows * n_per_row));

    size_t total_out = 0;
    for (int64_t row = 0; row < nrows; row += chunk_rows) {
        const int64_t rows_this = std::min<int64_t>(chunk_rows, nrows - row);
        const size_t chunk_bytes = static_cast<size_t>(rows_this) * dst_row_size;

        const int use_threads = std::max(1, static_cast<int>(std::min<int64_t>(nthread, rows_this)));
        if (use_threads == 1) {
            convert_rows_to_f32(tensor, row, rows_this, f32_chunk.data());
            const size_t got = ggml_quantize_chunk(
                dst_type,
                f32_chunk.data(),
                q_chunk.data(),
                0,
                rows_this,
                n_per_row,
                nullptr);
            if (got != chunk_bytes) {
                throw std::runtime_error("quantized chunk size mismatch");
            }
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(use_threads));
            std::vector<std::exception_ptr> errs(static_cast<size_t>(use_threads));

            for (int ti = 0; ti < use_threads; ++ti) {
                const int64_t r0 = (rows_this * ti) / use_threads;
                const int64_t r1 = (rows_this * (ti + 1)) / use_threads;
                if (r0 >= r1) {
                    continue;
                }
                workers.emplace_back([&, ti, r0, r1]() {
                    try {
                        const int64_t local_rows = r1 - r0;
                        std::vector<float> local_f32(static_cast<size_t>(local_rows * n_per_row));
                        convert_rows_to_f32(tensor, row + r0, local_rows, local_f32.data());

                        uint8_t * dst_ptr = q_chunk.data() + static_cast<size_t>(r0) * dst_row_size;
                        const size_t got = ggml_quantize_chunk(
                            dst_type,
                            local_f32.data(),
                            dst_ptr,
                            0,
                            local_rows,
                            n_per_row,
                            nullptr);
                        const size_t expected = static_cast<size_t>(local_rows) * dst_row_size;
                        if (got != expected) {
                            throw std::runtime_error("threaded quantized chunk size mismatch");
                        }
                    } catch (...) {
                        errs[static_cast<size_t>(ti)] = std::current_exception();
                    }
                });
            }

            for (auto & w : workers) {
                w.join();
            }
            for (const auto & e : errs) {
                if (e) {
                    std::rethrow_exception(e);
                }
            }
        }

        fout.write(reinterpret_cast<const char *>(q_chunk.data()), static_cast<std::streamsize>(chunk_bytes));
        total_out += chunk_bytes;
    }

    return total_out;
}

void print_usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s model-in.gguf model-out.gguf type [nthreads]\n", argv0);
    std::fprintf(stderr, "supported types:\n");
    std::fprintf(stderr, "  Q4_0\n");
    std::fprintf(stderr, "  Q4_1\n");
    std::fprintf(stderr, "  Q5_0\n");
    std::fprintf(stderr, "  Q5_1\n");
    std::fprintf(stderr, "  Q8_0\n");
    std::fprintf(stderr, "  Q2_K\n");
    std::fprintf(stderr, "  Q3_K\n");
    std::fprintf(stderr, "  Q4_K\n");
    std::fprintf(stderr, "  Q5_K\n");
    std::fprintf(stderr, "  Q6_K\n");
    std::fprintf(stderr, "  Q4_K_M\n");
}

int quantize_voxtral_gguf(
    const std::string & fname_inp,
    const std::string & fname_out,
    const quant_options & qopts,
    int nthread) {
    ggml_context * data_ctx_in = nullptr;
    gguf_context * gguf_in = nullptr;
    gguf_context * gguf_out = nullptr;

    try {
        gguf_init_params params{};
        params.no_alloc = false;
        params.ctx = &data_ctx_in;

        gguf_in = gguf_init_from_file(fname_inp.c_str(), params);
        if (gguf_in == nullptr || data_ctx_in == nullptr) {
            throw std::runtime_error("failed to load input GGUF");
        }

        const std::string arch = get_kv_str(gguf_in, "general.architecture");
        if (arch != kVoxtralArch && arch != "voxtral") {
            throw std::runtime_error(
                "unsupported architecture '" + arch + "', expected '" +
                std::string(kVoxtralArch) + "' or 'voxtral'");
        }

        gguf_out = gguf_init_empty();
        if (gguf_out == nullptr) {
            throw std::runtime_error("failed to create output GGUF context");
        }

        gguf_set_kv(gguf_out, gguf_in);
        gguf_set_val_u32(gguf_out, "general.quantization_version", GGML_QNT_VERSION);
        gguf_set_val_u32(gguf_out, "general.file_type", qopts.file_type);

        std::vector<tensor_ref> tensors;
        const int64_t n_tensors = gguf_get_n_tensors(gguf_in);
        tensors.reserve(static_cast<size_t>(n_tensors));

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name_c = gguf_get_tensor_name(gguf_in, i);
            if (name_c == nullptr) {
                throw std::runtime_error("encountered unnamed tensor in GGUF");
            }
            ggml_tensor * t = ggml_get_tensor(data_ctx_in, name_c);
            if (t == nullptr) {
                throw std::runtime_error("failed to resolve tensor data for '" + std::string(name_c) + "'");
            }
            tensors.push_back({std::string(name_c), t});
            gguf_add_tensor(gguf_out, t);
        }

        const size_t meta_size_placeholder = gguf_get_meta_size(gguf_out);
        const size_t alignment = gguf_get_alignment(gguf_out);

        std::ofstream fout(fname_out, std::ios::binary);
        if (!fout) {
            throw std::runtime_error("failed to open output file for writing");
        }
        fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        write_zeros(fout, meta_size_placeholder);

        size_t total_size_org = 0;
        size_t total_size_new = 0;
        int n_quantized = 0;

        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto & tw = tensors[i];
            ggml_tensor * tensor = tw.tensor;
            const std::string & name = tw.name;

            total_size_org += ggml_nbytes(tensor);

            bool quantize = is_voxtral_quantizable(name, tensor);
            ggml_type new_type = tensor->type;
            if (quantize) {
                new_type = choose_tensor_type(qopts, name);

                if (!ggml_is_quantized(new_type)) {
                    quantize = false;
                    new_type = tensor->type;
                }
                if (quantize && !is_row_compatible(new_type, tensor->ne[0])) {
                    std::fprintf(stderr,
                        "%s: warning: tensor '%s' width %" PRId64 " is incompatible with %s, leaving as %s\n",
                        __func__,
                        name.c_str(),
                        tensor->ne[0],
                        ggml_type_name(new_type),
                        ggml_type_name(tensor->type));
                    quantize = false;
                    new_type = tensor->type;
                }
                if (quantize && tensor->type == new_type) {
                    quantize = false;
                }
            }

            gguf_set_tensor_type(gguf_out, name.c_str(), new_type);
            const int64_t out_tid = gguf_find_tensor(gguf_out, name.c_str());
            if (out_tid < 0) {
                throw std::runtime_error("output GGUF lost tensor '" + name + "'");
            }

            std::fprintf(stdout,
                "[%4zu/%4zu] %36s - [%s], type = %6s, ",
                i + 1,
                tensors.size(),
                name.c_str(),
                tensor_shape_str(tensor).c_str(),
                ggml_type_name(tensor->type));

            size_t written = 0;
            if (!quantize) {
                written = ggml_nbytes(tensor);
                const size_t expected = tensor_nbytes_for_type(tensor, new_type);
                if (written != expected) {
                    throw std::runtime_error("tensor size mismatch while copying '" + name + "'");
                }
                fout.write(reinterpret_cast<const char *>(tensor->data), static_cast<std::streamsize>(written));
                std::fprintf(stdout, "size = %8.2f MiB\n", written / 1024.0 / 1024.0);
            } else {
                std::fprintf(stdout, "to %6s .. ", ggml_type_name(new_type));
                written = quantize_tensor_rows(tensor, new_type, nthread, fout);
                const size_t expected = tensor_nbytes_for_type(tensor, new_type);
                if (written != expected) {
                    throw std::runtime_error("tensor size mismatch after quantization for '" + name + "'");
                }
                std::fprintf(stdout,
                    "size = %8.2f MiB -> %8.2f MiB\n",
                    ggml_nbytes(tensor) / 1024.0 / 1024.0,
                    written / 1024.0 / 1024.0);
                ++n_quantized;
            }

            total_size_new += written;
            const size_t pad = (alignment - (written % alignment)) % alignment;
            write_zeros(fout, pad);
        }

        const size_t meta_size_final = gguf_get_meta_size(gguf_out);
        if (meta_size_final != meta_size_placeholder) {
            throw std::runtime_error("GGUF metadata size changed while quantizing");
        }

        std::vector<uint8_t> meta(meta_size_final);
        gguf_get_meta_data(gguf_out, meta.data());
        fout.seekp(0);
        fout.write(reinterpret_cast<const char *>(meta.data()), static_cast<std::streamsize>(meta.size()));
        fout.close();

        std::fprintf(stdout, "%s: quantized tensors = %d / %zu\n", __func__, n_quantized, tensors.size());
        std::fprintf(stdout, "%s: model size  = %8.2f MiB\n", __func__, total_size_org / 1024.0 / 1024.0);
        std::fprintf(stdout, "%s: quant size  = %8.2f MiB\n", __func__, total_size_new / 1024.0 / 1024.0);
    } catch (...) {
        if (gguf_out != nullptr) {
            gguf_free(gguf_out);
            gguf_out = nullptr;
        }
        if (gguf_in != nullptr) {
            gguf_free(gguf_in);
            gguf_in = nullptr;
        }
        if (data_ctx_in != nullptr) {
            ggml_free(data_ctx_in);
            data_ctx_in = nullptr;
        }
        throw;
    }

    if (gguf_out != nullptr) {
        gguf_free(gguf_out);
    }
    if (gguf_in != nullptr) {
        gguf_free(gguf_in);
    }
    if (data_ctx_in != nullptr) {
        ggml_free(data_ctx_in);
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    quant_options qopts;
    try {
        qopts = parse_quant_type(argv[3]);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }

    int nthread = static_cast<int>(std::thread::hardware_concurrency());
    if (nthread <= 0) {
        nthread = 1;
    }
    if (argc == 5) {
        try {
            nthread = std::stoi(argv[4]);
            if (nthread <= 0) {
                nthread = 1;
            }
        } catch (...) {
            std::fprintf(stderr, "invalid nthreads value: %s\n", argv[4]);
            return 1;
        }
    }

    const int64_t t_start_us = ggml_time_us();
    try {
        quantize_voxtral_gguf(fname_inp, fname_out, qopts, nthread);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: failed: %s\n", __func__, e.what());
        return 1;
    }
    const int64_t t_end_us = ggml_time_us();

    std::fprintf(stdout, "\n");
    std::fprintf(stdout, "%s: total time = %8.2f ms\n", __func__, (t_end_us - t_start_us) / 1000.0);
    return 0;
}
