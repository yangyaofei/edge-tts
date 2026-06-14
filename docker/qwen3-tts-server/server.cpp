/* server.cpp — HTTP server wrapping qwen3-tts.cpp
 *
 * Loads models once at startup and serves synthesis requests via a
 * simple JSON-over-HTTP API.
 *
 * Endpoints:
 *   GET  /api/health       -> {"status": "ok"}
 *   POST /api/synthesize   -> audio/wav
 *
 * Build: see CMakeLists.txt (links against the qwen3_tts static library).
 */

#include "qwen3_tts.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Utility: base64 decode
// ---------------------------------------------------------------------------
static inline int b64_char_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static std::vector<uint8_t> base64_decode(const std::string & input) {
    std::vector<uint8_t> output;
    output.reserve((input.size() / 4) * 3);

    int val = 0;
    int valb = -8;
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        if (c == '=') break;

        int d = b64_char_value(c);
        if (d < 0) continue;

        val = (val << 6) | d;
        valb += 6;
        if (valb >= 0) {
            output.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Utility: language string -> codec language id
// ---------------------------------------------------------------------------
static int32_t language_to_id(const std::string & lang) {
    if (lang == "en" || lang == "english")     return 2050;
    if (lang == "ru" || lang == "russian")     return 2069;
    if (lang == "zh" || lang == "chinese")     return 2055;
    if (lang == "ja" || lang == "japanese")    return 2058;
    if (lang == "ko" || lang == "korean")      return 2064;
    if (lang == "de" || lang == "german")      return 2053;
    if (lang == "fr" || lang == "french")      return 2061;
    if (lang == "es" || lang == "spanish")     return 2054;
    if (lang == "it" || lang == "italian")     return 2070;
    if (lang == "pt" || lang == "portuguese")  return 2071;
    // Default to English
    return 2050;
}

// ---------------------------------------------------------------------------
// Utility: generate a unique temp file path
// ---------------------------------------------------------------------------
static std::string make_temp_path(const std::string & suffix) {
    static std::atomic<uint64_t> counter{0};

    std::random_device rd;
    std::mt19937_64 gen(rd());
    uint64_t a = gen();
    uint64_t b = counter.fetch_add(1);

    char buf[256];
    snprintf(buf, sizeof(buf), "/tmp/qwen3tts_%016lx_%016lx%s",
             (unsigned long)a, (unsigned long)b, suffix.c_str());
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Utility: read entire file into a byte buffer
// ---------------------------------------------------------------------------
static bool read_file_bytes(const std::string & path, std::string & out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    out.assign(std::istreambuf_iterator<char>(ifs),
               std::istreambuf_iterator<char>());
    return true;
}

static bool write_file_bytes(const std::string & path,
                             const uint8_t * data, size_t len) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(len));
    return ofs.good();
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void print_usage(const char * program) {
    fprintf(stderr, "Usage: %s --model-dir <dir> [--port <n>] [--threads <n>]\n", program);
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --model-dir <dir>   Directory containing the GGUF models (required)\n");
    fprintf(stderr, "  --port <n>          HTTP listen port (default: 9880)\n");
    fprintf(stderr, "  --threads <n>       Number of CPU threads (default: %u)\n",
            (unsigned) std::thread::hardware_concurrency());
    fprintf(stderr, "  --host <addr>       Listen address (default: 0.0.0.0)\n");
    fprintf(stderr, "  -h, --help          Show this help\n");
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char ** argv) {
    std::string model_dir;
    int  port       = 9880;
    int  n_threads  = (int) std::thread::hardware_concurrency();
    if (n_threads <= 0) n_threads = 4;
    std::string host = "0.0.0.0";

    // ---- Parse arguments ----
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model-dir") {
            model_dir = need_value("--model-dir");
        } else if (arg == "--port") {
            port = std::atoi(need_value("--port"));
        } else if (arg == "--threads") {
            n_threads = std::atoi(need_value("--threads"));
        } else if (arg == "--host") {
            host = need_value("--host");
        } else {
            fprintf(stderr, "Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_dir.empty()) {
        fprintf(stderr, "Error: --model-dir is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load models once at startup ----
    fprintf(stderr, "=== qwen3-tts-server ===\n");
    fprintf(stderr, "Model dir : %s\n", model_dir.c_str());
    fprintf(stderr, "Threads   : %d\n", n_threads);

    qwen3_tts::Qwen3TTS tts;

    fprintf(stderr, "Loading models...\n");
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: failed to load models: %s\n", tts.get_error().c_str());
        return 1;
    }
    if (!tts.is_loaded()) {
        fprintf(stderr, "Error: models not loaded after load_models()\n");
        return 1;
    }
    fprintf(stderr, "Models loaded successfully.\n");

    // Progress callback (best-effort, writes to stderr)
    tts.set_progress_callback([](int tokens, int max_tokens) {
        fprintf(stderr, "\r  generating: %d/%d tokens", tokens, max_tokens);
    });

    // Serialize all synthesis requests — the pipeline has mutable internal
    // state (KV cache, backend scheduler) that is not thread-safe.
    std::mutex tts_mutex;

    // ---- HTTP server ----
    httplib::Server svr;

    // Keep-alive timeout tuning
    svr.set_keep_alive_max_count(16);
    svr.set_read_timeout (300);  // seconds — synthesis can be slow
    svr.set_write_timeout(300);

    // ---- GET /api/health ----
    svr.Get("/api/health",
        [&tts](const httplib::Request &, httplib::Response & res) {
            json j;
            j["status"]      = tts.is_loaded() ? "ok" : "loading";
            j["model_loaded"] = tts.is_loaded();
            res.set_content(j.dump(), "application/json");
        });

    // ---- POST /api/synthesize ----
    svr.Post("/api/synthesize",
        [&](const httplib::Request & req, httplib::Response & res) {
            // -- Parse JSON body --
            json body;
            try {
                body = json::parse(req.body);
            } catch (const std::exception & e) {
                res.status = 400;
                json err;
                err["error"] = std::string("Invalid JSON: ") + e.what();
                res.set_content(err.dump(), "application/json");
                return;
            }

            // -- Validate required 'text' --
            if (!body.contains("text") || !body["text"].is_string()) {
                res.status = 400;
                json err;
                err["error"] = "Missing required field: 'text' (string)";
                res.set_content(err.dump(), "application/json");
                return;
            }
            std::string text = body["text"].get<std::string>();
            if (text.empty()) {
                res.status = 400;
                json err;
                err["error"] = "'text' must not be empty";
                res.set_content(err.dump(), "application/json");
                return;
            }

            // -- Build params --
            qwen3_tts::tts_params params;
            params.n_threads    = n_threads;
            params.print_timing = false;

            if (body.contains("language") && body["language"].is_string()) {
                params.language_id = language_to_id(body["language"].get<std::string>());
            }
            if (body.contains("max_tokens") && body["max_tokens"].is_number()) {
                params.max_audio_tokens = body["max_tokens"].get<int32_t>();
            }
            if (body.contains("temperature") && body["temperature"].is_number()) {
                params.temperature = body["temperature"].get<float>();
            }
            if (body.contains("top_k") && body["top_k"].is_number()) {
                params.top_k = body["top_k"].get<int32_t>();
            }
            if (body.contains("top_p") && body["top_p"].is_number()) {
                params.top_p = body["top_p"].get<float>();
            }
            if (body.contains("repetition_penalty") && body["repetition_penalty"].is_number()) {
                params.repetition_penalty = body["repetition_penalty"].get<float>();
            }

            // -- Handle optional ref_audio (base64 -> temp WAV file) --
            std::string ref_temp_path;
            bool has_ref = false;

            if (body.contains("ref_audio") && body["ref_audio"].is_string()) {
                const std::string & b64 = body["ref_audio"].get<std::string>();
                if (!b64.empty()) {
                    auto decoded = base64_decode(b64);
                    if (decoded.empty()) {
                        res.status = 400;
                        json err;
                        err["error"] = "Failed to decode ref_audio base64";
                        res.set_content(err.dump(), "application/json");
                        return;
                    }
                    ref_temp_path = make_temp_path(".wav");
                    if (!write_file_bytes(ref_temp_path, decoded.data(), decoded.size())) {
                        res.status = 500;
                        json err;
                        err["error"] = "Failed to write reference audio temp file";
                        res.set_content(err.dump(), "application/json");
                        return;
                    }
                    has_ref = true;
                    fprintf(stderr, "  reference audio: %zu bytes -> %s\n",
                            decoded.size(), ref_temp_path.c_str());
                }
            }

            // -- Synthesize (thread-safe) --
            fprintf(stderr, "\n[req] synthesize (lang=%d, max_tokens=%d, voice_clone=%s): \"%s\"\n",
                    params.language_id, params.max_audio_tokens,
                    has_ref ? "yes" : "no",
                    text.substr(0, 80).c_str());

            qwen3_tts::tts_result result;
            {
                std::lock_guard<std::mutex> lock(tts_mutex);
                if (has_ref) {
                    result = tts.synthesize_with_voice(text, ref_temp_path, params);
                } else {
                    result = tts.synthesize(text, params);
                }
            }

            // -- Cleanup reference temp file --
            if (!ref_temp_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(ref_temp_path, ec);
            }

            // -- Check result --
            if (!result.success) {
                fprintf(stderr, "  [err] synthesis failed: %s\n", result.error_msg.c_str());
                res.status = 500;
                json err;
                err["error"] = "Synthesis failed: " + result.error_msg;
                res.set_content(err.dump(), "application/json");
                return;
            }

            fprintf(stderr, "\n  [ok] %zu samples (%.2f s) in %lld ms\n",
                    result.audio.size(),
                    (double) result.audio.size() / result.sample_rate,
                    (long long) result.t_total_ms);

            // -- Encode WAV to temp file, then read back into response --
            std::string wav_temp = make_temp_path(".wav");
            if (!qwen3_tts::save_audio_file(wav_temp, result.audio, result.sample_rate)) {
                std::error_code ec;
                std::filesystem::remove(wav_temp, ec);
                res.status = 500;
                json err;
                err["error"] = "Failed to encode WAV output";
                res.set_content(err.dump(), "application/json");
                return;
            }

            std::string wav_bytes;
            bool ok = read_file_bytes(wav_temp, wav_bytes);

            std::error_code ec;
            std::filesystem::remove(wav_temp, ec);

            if (!ok) {
                res.status = 500;
                json err;
                err["error"] = "Failed to read encoded WAV output";
                res.set_content(err.dump(), "application/json");
                return;
            }

            // -- Return audio --
            res.set_content(wav_bytes, "audio/wav");
        });

    // ---- Graceful shutdown logging ----
    svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "[%d] %s %s\n", res.status, req.method.c_str(), req.path.c_str());
    });

    // ---- Start listening ----
    fprintf(stderr, "\nListening on http://%s:%d\n", host.c_str(), port);
    fprintf(stderr, "  GET  /api/health\n");
    fprintf(stderr, "  POST /api/synthesize\n");

    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "Error: failed to listen on %s:%d\n", host.c_str(), port);
        return 1;
    }

    return 0;
}
