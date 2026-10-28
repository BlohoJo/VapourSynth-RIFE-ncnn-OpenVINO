/*
    MIT License

    Copyright (c) 2021-2022 HolyWu
    Copyright (c) 2026 BlohoJo (OpenVINO port)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/*
    VapourSynth-RIFE-OpenVINO
    -------------------------
    RIFE frame interpolation running on the OpenVINO runtime (CPU plugin by
    default). The VapourSynth side of this file - parameter semantics, frame
    mapping for factor_num/factor_den/fps_num/fps_den, scene-change handling
    via _SceneChangeNext, PSNR based skipping and the parallel request pool -
    follows the original ncnn-Vulkan plugin by HolyWu / styler00dollar as
    closely as practical. The inference backend lives in rife_ov.cpp.
*/

#include <algorithm>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "VapourSynth4.h"
#include "VSHelper4.h"

#include "rife_ov.h"

using namespace std::literals;

namespace rifeov {

static constexpr const char* PLUGIN_ID{ "com.blohojo.rifeov" };

/*
    Numbered model table. The indices are the ones used by the ncnn-Vulkan
    plugin so existing scripts keep working; `file` is the base name of the
    corresponding vs-mlrt ONNX export (without extension) or nullptr when no
    ONNX model exists for that index (rife v1/v2/v3 and HD/UHD/anime).
*/
struct ModelEntry {
    const char* ncnnName;
    const char* file;
};

static const ModelEntry MODEL_TABLE[]{
    /* 0  */ { "rife", nullptr },
    /* 1  */ { "rife-HD", nullptr },
    /* 2  */ { "rife-UHD", nullptr },
    /* 3  */ { "rife-anime", nullptr },
    /* 4  */ { "rife-v2", nullptr },
    /* 5  */ { "rife-v2.3", nullptr },
    /* 6  */ { "rife-v2.4", nullptr },
    /* 7  */ { "rife-v3.0", nullptr },
    /* 8  */ { "rife-v3.1", nullptr },
    /* 9  */ { "rife-v3.9_ensembleFalse_fastTrue", nullptr },
    /* 10 */ { "rife-v3.9_ensembleTrue_fastFalse", nullptr },
    /* 11 */ { "rife-v4_ensembleFalse_fastTrue", "rife_v4.0" },
    /* 12 */ { "rife-v4_ensembleTrue_fastFalse", "rife_v4.0_ensemble" },
    /* 13 */ { "rife-v4.1_ensembleFalse_fastTrue", nullptr },   // vs-mlrt has no v4.1 export
    /* 14 */ { "rife-v4.1_ensembleTrue_fastFalse", nullptr },
    /* 15 */ { "rife-v4.2_ensembleFalse_fastTrue", "rife_v4.2" },
    /* 16 */ { "rife-v4.2_ensembleTrue_fastFalse", "rife_v4.2_ensemble" },
    /* 17 */ { "rife-v4.3_ensembleFalse_fastTrue", "rife_v4.3" },
    /* 18 */ { "rife-v4.3_ensembleTrue_fastFalse", "rife_v4.3_ensemble" },
    /* 19 */ { "rife-v4.4_ensembleFalse_fastTrue", "rife_v4.4" },
    /* 20 */ { "rife-v4.4_ensembleTrue_fastFalse", "rife_v4.4_ensemble" },
    /* 21 */ { "rife-v4.5_ensembleFalse", "rife_v4.5" },
    /* 22 */ { "rife-v4.5_ensembleTrue", "rife_v4.5_ensemble" },
    /* 23 */ { "rife-v4.6_ensembleFalse", "rife_v4.6" },
    /* 24 */ { "rife-v4.6_ensembleTrue", "rife_v4.6_ensemble" },
    /* 25 */ { "rife-v4.7_ensembleFalse", "rife_v4.7" },
    /* 26 */ { "rife-v4.7_ensembleTrue", "rife_v4.7_ensemble" },
    /* 27 */ { "rife-v4.8_ensembleFalse", "rife_v4.8" },
    /* 28 */ { "rife-v4.8_ensembleTrue", "rife_v4.8_ensemble" },
    /* 29 */ { "rife-v4.9_ensembleFalse", "rife_v4.9" },
    /* 30 */ { "rife-v4.9_ensembleTrue", "rife_v4.9_ensemble" },
    /* 31 */ { "rife-v4.10_ensembleFalse", "rife_v4.10" },
    /* 32 */ { "rife-v4.10_ensembleTrue", "rife_v4.10_ensemble" },
    /* 33 */ { "rife-v4.11_ensembleFalse", "rife_v4.11" },
    /* 34 */ { "rife-v4.11_ensembleTrue", "rife_v4.11_ensemble" },
    /* 35 */ { "rife-v4.12_ensembleFalse", "rife_v4.12" },
    /* 36 */ { "rife-v4.12_ensembleTrue", "rife_v4.12_ensemble" },
    /* 37 */ { "rife-v4.12_lite_ensembleFalse", "rife_v4.12_lite" },
    /* 38 */ { "rife-v4.12_lite_ensembleTrue", "rife_v4.12_lite_ensemble" },
    /* 39 */ { "rife-v4.13_ensembleFalse", "rife_v4.13" },
    /* 40 */ { "rife-v4.13_ensembleTrue", "rife_v4.13_ensemble" },
    /* 41 */ { "rife-v4.13_lite_ensembleFalse", "rife_v4.13_lite" },
    /* 42 */ { "rife-v4.13_lite_ensembleTrue", "rife_v4.13_lite_ensemble" },
    /* 43 */ { "rife-v4.14_ensembleFalse", "rife_v4.14" },
    /* 44 */ { "rife-v4.14_ensembleTrue", "rife_v4.14_ensemble" },
    /* 45 */ { "rife-v4.14_lite_ensembleFalse", "rife_v4.14_lite" },
    /* 46 */ { "rife-v4.14_lite_ensembleTrue", "rife_v4.14_lite_ensemble" },
    /* 47 */ { "rife-v4.15_ensembleFalse", "rife_v4.15" },
    /* 48 */ { "rife-v4.15_ensembleTrue", "rife_v4.15_ensemble" },
    /* 49 */ { "rife-v4.15_lite_ensembleFalse", "rife_v4.15_lite" },
    /* 50 */ { "rife-v4.15_lite_ensembleTrue", "rife_v4.15_lite_ensemble" },
    /* 51 */ { "rife-v4.16_lite_ensembleFalse", "rife_v4.16_lite" },
    /* 52 */ { "rife-v4.16_lite_ensembleTrue", "rife_v4.16_lite_ensemble" },
    /* 53 */ { "rife-v4.17_ensembleFalse", "rife_v4.17" },
    /* 54 */ { "rife-v4.17_ensembleTrue", "rife_v4.17_ensemble" },
    /* 55 */ { "rife-v4.17_lite_ensembleFalse", "rife_v4.17_lite" },
    /* 56 */ { "rife-v4.17_lite_ensembleTrue", "rife_v4.17_lite_ensemble" },
    /* 57 */ { "rife-v4.18_ensembleFalse", "rife_v4.18" },
    /* 58 */ { "rife-v4.18_ensembleTrue", "rife_v4.18_ensemble" },
    /* 59 */ { "rife-v4.19_beta_ensembleFalse", "rife_v4.19" },
    /* 60 */ { "rife-v4.19_beta_ensembleTrue", "rife_v4.19_ensemble" },
    /* 61 */ { "rife-v4.20_ensembleFalse", "rife_v4.20" },
    /* 62 */ { "rife-v4.20_ensembleTrue", "rife_v4.20_ensemble" },
    /* 63 */ { "rife-v4.21_ensembleFalse", "rife_v4.21" },
    /* 64 */ { "rife-v4.22_ensembleFalse", "rife_v4.22" },
    /* 65 */ { "rife-v4.22_lite_ensembleFalse", "rife_v4.22_lite" },
    /* 66 */ { "rife-v4.23_beta_ensembleFalse", "rife_v4.23" },
    /* 67 */ { "rife-v4.24_ensembleFalse", "rife_v4.24" },
    /* 68 */ { "rife-v4.24_ensembleTrue", "rife_v4.24_ensemble" },
    /* 69 */ { "rife-v4.25_ensembleFalse", "rife_v4.25" },
    /* 70 */ { "rife-v4.25-lite_ensembleFalse", "rife_v4.25_lite" },
    /* 71 */ { "rife-v4.25_heavy_beta_ensembleFalse", "rife_v4.25_heavy" },
    /* 72 */ { "rife-v4.26_ensembleFalse", "rife_v4.26" },
    /* 73 */ { "rife-v4.26-large_ensembleFalse", "rife_v4.26_heavy" },
};
static constexpr int MODEL_COUNT{ static_cast<int>(sizeof(MODEL_TABLE) / sizeof(MODEL_TABLE[0])) };
static constexpr int DEFAULT_MODEL{ 72 }; // rife_v4.26

/*
    Padding multiple required by a model, derived from its file name using the
    same rules as vs-mlrt's tilesize_requirement():
      v4.26 and newer -> 64, v4.25_lite -> 128, v4.25_heavy -> 64, else 32.
    Unknown names fall back to 32 (the user can override with `pad`).
*/
static int detectPadding(const std::string& fileName) {
    std::string name{ fileName };
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    static const std::regex re{ R"(v(\d+)[._](\d+))" };
    std::smatch m;
    if (!std::regex_search(name, m, re))
        return 32;

    const int major{ std::stoi(m[1]) };
    const int minor{ std::stoi(m[2]) };
    const bool lite{ name.find("lite") != std::string::npos };
    const bool heavy{ name.find("heavy") != std::string::npos || name.find("large") != std::string::npos };

    if (major > 4 || (major == 4 && minor >= 26))
        return 64;
    if (major == 4 && minor == 25 && lite)
        return 128;
    if (major == 4 && minor == 25 && heavy)
        return 64;
    return 32;
}

struct RIFEData final {
    VSNode* node{};
    VSNode* psnr{};
    VSVideoInfo vi{};
    bool sceneChange{};
    bool skip{};
    double skipThreshold{};
    int64_t factor{};
    int64_t factorNum{};
    int64_t factorDen{};
    std::unique_ptr<RIFE> rife;
};

static void filter(const VSFrame* src0, const VSFrame* src1, VSFrame* dst,
                   const float timestep, const RIFEData* const VS_RESTRICT d, const VSAPI* vsapi) {
    const auto width{ vsapi->getFrameWidth(src0, 0) };
    const auto height{ vsapi->getFrameHeight(src0, 0) };
    const auto stride{ vsapi->getStride(src0, 0) / d->vi.format.bytesPerSample };
    auto src0R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 0)) };
    auto src0G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 1)) };
    auto src0B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src0, 2)) };
    auto src1R{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 0)) };
    auto src1G{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 1)) };
    auto src1B{ reinterpret_cast<const float*>(vsapi->getReadPtr(src1, 2)) };
    auto dstR{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)) };
    auto dstG{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)) };
    auto dstB{ reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)) };

    // The request pool inside RIFE blocks when all `threads` requests are busy
    // (equivalent of the gpu_thread semaphore in the original plugin).
    d->rife->process(src0R, src0G, src0B, src1R, src1G, src1B, dstR, dstG, dstB, width, height, stride, timestep);
}

static const VSFrame* VS_CC rifeGetFrame(int n, int activationReason, void* instanceData, [[maybe_unused]] void** frameData,
                                         VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<const RIFEData*>(instanceData) };

    auto frameNum{ static_cast<int>(n * d->factorDen / d->factorNum) };
    auto remainder{ n * d->factorDen % d->factorNum };

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(frameNum, d->node, frameCtx);
        if (remainder != 0 && n < d->vi.numFrames - d->factor)
            vsapi->requestFrameFilter(frameNum + 1, d->node, frameCtx);

        if (d->skip)
            vsapi->requestFrameFilter(frameNum, d->psnr, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        auto src0{ vsapi->getFrameFilter(frameNum, d->node, frameCtx) };
        decltype(src0) src1{};
        decltype(src0) psnr{};
        VSFrame* dst{};

        if (remainder != 0 && n < d->vi.numFrames - d->factor) {
            bool sceneChange{};
            double psnrY{ -1.0 };
            int err;

            if (d->sceneChange)
                sceneChange = !!vsapi->mapGetInt(vsapi->getFramePropertiesRO(src0), "_SceneChangeNext", 0, &err);

            if (d->skip) {
                psnr = vsapi->getFrameFilter(frameNum, d->psnr, frameCtx);
                psnrY = vsapi->mapGetFloat(vsapi->getFramePropertiesRO(psnr), "psnr_y", 0, nullptr);
            }

            if (sceneChange || psnrY >= d->skipThreshold) {
                dst = vsapi->copyFrame(src0, core);
            } else {
                src1 = vsapi->getFrameFilter(frameNum + 1, d->node, frameCtx);
                dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src0, core);
                try {
                    filter(src0, src1, dst, static_cast<float>(remainder) / d->factorNum, d, vsapi);
                } catch (const std::exception& e) {
                    vsapi->setFilterError(("RIFE: "s + e.what()).c_str(), frameCtx);
                    vsapi->freeFrame(src0);
                    vsapi->freeFrame(src1);
                    vsapi->freeFrame(psnr);
                    vsapi->freeFrame(dst);
                    return nullptr;
                }
            }
        } else {
            dst = vsapi->copyFrame(src0, core);
        }

        auto props{ vsapi->getFramePropertiesRW(dst) };
        int errNum, errDen;
        auto durationNum{ vsapi->mapGetInt(props, "_DurationNum", 0, &errNum) };
        auto durationDen{ vsapi->mapGetInt(props, "_DurationDen", 0, &errDen) };
        if (!errNum && !errDen) {
            vsh::muldivRational(&durationNum, &durationDen, d->factorDen, d->factorNum);
            vsapi->mapSetInt(props, "_DurationNum", durationNum, maReplace);
            vsapi->mapSetInt(props, "_DurationDen", durationDen, maReplace);
        }

        vsapi->freeFrame(src0);
        vsapi->freeFrame(src1);
        vsapi->freeFrame(psnr);
        return dst;
    }

    return nullptr;
}

static void VS_CC rifeFree(void* instanceData, [[maybe_unused]] VSCore* core, const VSAPI* vsapi) {
    auto d{ static_cast<RIFEData*>(instanceData) };
    vsapi->freeNode(d->node);
    vsapi->freeNode(d->psnr);
    delete d;
}

// Helper: invoke a function of another plugin and propagate its error.
static VSMap* invokeOrThrow(const VSAPI* vsapi, VSPlugin* plugin, const char* name, VSMap* args, std::string& error) {
    auto ret{ vsapi->invoke(plugin, name, args) };
    if (auto e{ vsapi->mapGetError(ret) }) {
        error = e;
        vsapi->freeMap(ret);
        return nullptr;
    }
    return ret;
}

static std::filesystem::path toPath(const std::string& utf8) {
#ifdef _WIN32
    return std::filesystem::path(reinterpret_cast<const char8_t*>(utf8.c_str()));
#else
    return std::filesystem::path(utf8);
#endif
}

static std::string fromPath(const std::filesystem::path& p) {
#ifdef _WIN32
    auto u8{ p.u8string() };
    return std::string(u8.begin(), u8.end());
#else
    return p.string();
#endif
}

/*
    Resolve the model file to load.
      * model_path empty          -> <plugin dir>/models/<table name>.onnx (or .xml)
      * model_path is a directory -> <model_path>/<table name>.onnx (or .xml)
      * model_path is a file      -> used as-is (`model` is ignored)
*/
static std::filesystem::path resolveModelFile(const std::string& modelPath, int model, const VSAPI* vsapi, VSCore* core) {
    std::filesystem::path base;
    bool isFile{};

    if (modelPath.empty()) {
        std::string pluginPath{ vsapi->getPluginPath(vsapi->getPluginByID(PLUGIN_ID, core)) };
        base = toPath(pluginPath).parent_path() / "models";
    } else {
        base = toPath(modelPath);
        std::error_code ec;
        isFile = std::filesystem::is_regular_file(base, ec);
        if (!isFile && !std::filesystem::is_directory(base, ec))
            throw std::runtime_error("model_path '" + modelPath + "' is neither a file nor a directory");
    }

    if (isFile)
        return base;

    if (model < 0 || model >= MODEL_COUNT)
        throw std::runtime_error("model must be between 0 and " + std::to_string(MODEL_COUNT - 1) + " (inclusive)");

    const auto& entry{ MODEL_TABLE[model] };
    if (!entry.file)
        throw std::runtime_error("model " + std::to_string(model) + " (" + entry.ncnnName +
                                 ") has no ONNX/OpenVINO equivalent; only RIFE v4.x models are available "
                                 "(see README for the list of supported indices)");

    std::error_code ec;
    for (const char* ext : { ".onnx", ".xml" }) {
        auto candidate{ base / (std::string{ entry.file } + ext) };
        if (std::filesystem::is_regular_file(candidate, ec))
            return candidate;
    }
    throw std::runtime_error("model file '" + fromPath(base / (std::string{ entry.file } + ".onnx")) +
                             "' (or .xml) not found. Download it with scripts/download_models.py "
                             "or set model_path");
}

static void VS_CC rifeCreate(const VSMap* in, VSMap* out, [[maybe_unused]] void* userData, VSCore* core, const VSAPI* vsapi) {
    auto d{ std::make_unique<RIFEData>() };

    try {
        d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
        d->vi = *vsapi->getVideoInfo(d->node);
        int err;

        if (!vsh::isConstantVideoFormat(&d->vi) ||
            d->vi.format.colorFamily != cfRGB ||
            d->vi.format.sampleType != stFloat ||
            d->vi.format.bitsPerSample != 32)
            throw std::runtime_error("only constant RGB format 32 bit float input supported");

        auto model{ vsapi->mapGetIntSaturated(in, "model", 0, &err) };
        if (err)
            model = DEFAULT_MODEL;

        auto factorNum{ vsapi->mapGetInt(in, "factor_num", 0, &err) };
        if (err)
            factorNum = 2;

        auto factorDen{ vsapi->mapGetInt(in, "factor_den", 0, &err) };
        if (err)
            factorDen = 1;

        auto fpsNum{ vsapi->mapGetInt(in, "fps_num", 0, &err) };
        if (!err && fpsNum < 1)
            throw std::runtime_error("fps_num must be at least 1");

        auto fpsDen{ vsapi->mapGetInt(in, "fps_den", 0, &err) };
        if (!err && fpsDen < 1)
            throw std::runtime_error("fps_den must be at least 1");

        auto model_path{ vsapi->mapGetData(in, "model_path", 0, &err) };
        std::string modelPath{ err ? "" : model_path };

        auto device{ vsapi->mapGetData(in, "device", 0, &err) };
        std::string deviceName{ err ? "CPU" : device };

        auto threads{ vsapi->mapGetIntSaturated(in, "threads", 0, &err) };
        if (err)
            threads = 1;

        auto ovThreads{ vsapi->mapGetIntSaturated(in, "ov_threads", 0, &err) };
        if (err)
            ovThreads = 0;

        auto precision{ vsapi->mapGetData(in, "precision", 0, &err) };
        std::string precisionStr{ err ? "f32" : precision };

        auto hint{ vsapi->mapGetData(in, "hint", 0, &err) };
        std::string hintStr{ err ? "" : hint };

        auto cache_dir{ vsapi->mapGetData(in, "cache_dir", 0, &err) };
        std::string cacheDir{ err ? "" : cache_dir };

        auto pad{ vsapi->mapGetIntSaturated(in, "pad", 0, &err) };
        if (err)
            pad = 0; // auto

        d->sceneChange = !!vsapi->mapGetInt(in, "sc", 0, &err);
        d->skip = !!vsapi->mapGetInt(in, "skip", 0, &err);

        d->skipThreshold = vsapi->mapGetFloat(in, "skip_threshold", 0, &err);
        if (err)
            d->skipThreshold = 60.0;

        if (model < 0 || model >= MODEL_COUNT)
            throw std::runtime_error("model must be between 0 and " + std::to_string(MODEL_COUNT - 1) + " (inclusive)");

        if (factorNum < 1)
            throw std::runtime_error("factor_num must be at least 1");

        if (factorDen < 1)
            throw std::runtime_error("factor_den must be at least 1");

        if (fpsNum && fpsDen && !(d->vi.fpsNum && d->vi.fpsDen))
            throw std::runtime_error("clip does not have a valid frame rate and hence fps_num and fps_den cannot be used");

        if (threads < 1)
            throw std::runtime_error("threads must be greater than 0");

        if (ovThreads < 0)
            throw std::runtime_error("ov_threads must be 0 (auto) or greater");

        if (pad < 0)
            throw std::runtime_error("pad must be 0 (auto) or a positive multiple");

        if (d->skipThreshold < 0 || d->skipThreshold > 60)
            throw std::runtime_error("skip_threshold must be between 0.0 and 60.0 (inclusive)");

        if (fpsNum && fpsDen) {
            vsh::muldivRational(&fpsNum, &fpsDen, d->vi.fpsDen, d->vi.fpsNum);
            d->factorNum = fpsNum;
            d->factorDen = fpsDen;
        } else {
            d->factorNum = factorNum;
            d->factorDen = factorDen;
        }
        vsh::muldivRational(&d->vi.fpsNum, &d->vi.fpsDen, d->factorNum, d->factorDen);

        if (d->vi.numFrames < 2)
            throw std::runtime_error("clip's number of frames must be at least 2");

        if (d->vi.numFrames / d->factorDen > INT_MAX / d->factorNum)
            throw std::runtime_error("resulting clip is too long");

        auto oldNumFrames{ d->vi.numFrames };
        d->vi.numFrames = static_cast<int>(d->vi.numFrames * d->factorNum / d->factorDen);

        d->factor = d->factorNum / d->factorDen;

        // list_devices: print the OpenVINO devices onto the clip and return (analog of list_gpu).
        if (!!vsapi->mapGetInt(in, "list_devices", 0, &err)) {
            std::string text{ RIFE::listDevices() };

            auto args{ vsapi->createMap() };
            vsapi->mapConsumeNode(args, "clip", d->node, maReplace);
            d->node = nullptr;
            vsapi->mapSetData(args, "text", text.c_str(), -1, dtUtf8, maReplace);

            std::string error;
            auto ret{ invokeOrThrow(vsapi, vsapi->getPluginByID(VSH_TEXT_PLUGIN_ID, core), "Text", args, error) };
            vsapi->freeMap(args);
            if (!ret)
                throw std::runtime_error(error);

            vsapi->mapConsumeNode(out, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->freeMap(ret);
            return;
        }

        const auto modelFile{ resolveModelFile(modelPath, model, vsapi, core) };
        const std::string modelFileStr{ fromPath(modelFile) };

        int padding{ pad > 0 ? pad : detectPadding(fromPath(modelFile.filename())) };

        // skip=True: build a PSNR clip with the VMAF plugin exactly like the original plugin.
        if (d->skip) {
            auto vmaf{ vsapi->getPluginByID("com.holywu.vmaf", core) };
            if (!vmaf)
                throw std::runtime_error("VMAF plugin is required when skip=True");

            std::string error;
            auto args{ vsapi->createMap() };
            vsapi->mapSetNode(args, "clip", d->node, maReplace);
            vsapi->mapSetInt(args, "width", std::min(d->vi.width, 512), maReplace);
            vsapi->mapSetInt(args, "height", std::min(d->vi.height, 512), maReplace);
            vsapi->mapSetInt(args, "format", pfYUV420P8, maReplace);
            vsapi->mapSetData(args, "matrix_s", "709", -1, dtUtf8, maReplace);

            auto ret{ invokeOrThrow(vsapi, vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core), "Bicubic", args, error) };
            if (!ret) {
                vsapi->freeMap(args);
                throw std::runtime_error(error);
            }

            vsapi->clearMap(args);
            auto reference{ vsapi->mapGetNode(ret, "clip", 0, nullptr) };
            vsapi->mapSetNode(args, "clip", reference, maReplace);
            vsapi->mapSetInt(args, "frames", oldNumFrames - 1, maReplace);

            vsapi->freeMap(ret);
            ret = invokeOrThrow(vsapi, vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "DuplicateFrames", args, error);
            if (!ret) {
                vsapi->freeMap(args);
                vsapi->freeNode(reference);
                throw std::runtime_error(error);
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "clip", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "first", 1, maReplace);

            vsapi->freeMap(ret);
            ret = invokeOrThrow(vsapi, vsapi->getPluginByID(VSH_STD_PLUGIN_ID, core), "Trim", args, error);
            if (!ret) {
                vsapi->freeMap(args);
                vsapi->freeNode(reference);
                throw std::runtime_error(error);
            }

            vsapi->clearMap(args);
            vsapi->mapConsumeNode(args, "reference", reference, maReplace);
            vsapi->mapConsumeNode(args, "distorted", vsapi->mapGetNode(ret, "clip", 0, nullptr), maReplace);
            vsapi->mapSetInt(args, "feature", 0, maReplace);

            vsapi->freeMap(ret);
            ret = invokeOrThrow(vsapi, vmaf, "Metric", args, error);
            vsapi->freeMap(args);
            if (!ret)
                throw std::runtime_error(error);

            d->psnr = vsapi->mapGetNode(ret, "clip", 0, nullptr);
            vsapi->freeMap(ret);
        }

        RIFEOptions options;
        options.modelFile = modelFileStr;
        options.device = deviceName;
        options.width = d->vi.width;
        options.height = d->vi.height;
        options.padding = padding;
        options.numRequests = threads;
        options.ovThreads = ovThreads;
        options.precision = precisionStr;
        options.hint = hintStr;
        options.cacheDir = cacheDir;

        d->rife = std::make_unique<RIFE>(options);
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("RIFE: "s + e.what()).c_str());
        vsapi->freeNode(d->node);
        vsapi->freeNode(d->psnr);
        return;
    }

    std::vector<VSFilterDependency> deps{ { d->node, rpGeneral } };
    if (d->skip)
        deps.push_back({ d->psnr, rpGeneral });
    vsapi->createVideoFilter(out, "RIFE", &d->vi, rifeGetFrame, rifeFree, fmParallel, deps.data(), static_cast<int>(deps.size()), d.get(), core);
    d.release();
}

} // namespace rifeov

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(rifeov::PLUGIN_ID, "rifeov",
                         "Real-Time Intermediate Flow Estimation for Video Frame Interpolation (OpenVINO)",
                         VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("RIFE",
                             "clip:vnode;"
                             "model:int:opt;"
                             "factor_num:int:opt;"
                             "factor_den:int:opt;"
                             "fps_num:int:opt;"
                             "fps_den:int:opt;"
                             "model_path:data:opt;"
                             "device:data:opt;"
                             "threads:int:opt;"
                             "ov_threads:int:opt;"
                             "precision:data:opt;"
                             "hint:data:opt;"
                             "cache_dir:data:opt;"
                             "pad:int:opt;"
                             "sc:int:opt;"
                             "skip:int:opt;"
                             "skip_threshold:float:opt;"
                             "list_devices:int:opt;",
                             "clip:vnode;",
                             rifeov::rifeCreate, nullptr, plugin);
}
