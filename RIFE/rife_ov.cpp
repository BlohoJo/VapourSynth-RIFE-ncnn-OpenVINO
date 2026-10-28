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

#include "rife_ov.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace rifeov {

namespace {

// Read a model from a UTF-8 path. On Windows OpenVINO wants a wide string for
// non-ASCII paths, everywhere else std::string is fine.
std::shared_ptr<ov::Model> readModel(ov::Core& core, const std::string& utf8Path) {
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    auto bufferSize{ MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, nullptr, 0) };
    std::wstring wpath(static_cast<size_t>(bufferSize), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, wpath.data(), bufferSize);
    wpath.resize(wpath.size() - 1); // drop terminating NUL
    return core.read_model(wpath);
#else
    return core.read_model(utf8Path);
#endif
}

// Copy one RGB plane of the source frame into channel `channel` of a padded
// [C, H_p, W_p] float tensor, replicating the last column / row (edge padding),
// exactly like the ncnn preproc shader did with clamped sampling.
void copyPlanePadded(float* dst, const float* src, int width, int height, std::ptrdiff_t stride,
                     int paddedWidth, int paddedHeight) {
    for (int y = 0; y < height; y++) {
        const float* s{ src + y * stride };
        float* d{ dst + static_cast<size_t>(y) * paddedWidth };
        std::memcpy(d, s, sizeof(float) * width);
        const float edge{ s[width - 1] };
        for (int x = width; x < paddedWidth; x++)
            d[x] = edge;
    }
    for (int y = height; y < paddedHeight; y++) {
        std::memcpy(dst + static_cast<size_t>(y) * paddedWidth,
                    dst + static_cast<size_t>(height - 1) * paddedWidth,
                    sizeof(float) * paddedWidth);
    }
}

void fillPlane(float* dst, size_t count, float value) {
    std::fill_n(dst, count, value);
}

} // namespace

std::string RIFE::listDevices() {
    ov::Core core;
    std::string text;
    for (const auto& dev : core.get_available_devices()) {
        text += dev;
        try {
            text += ": " + core.get_property(dev, ov::device::full_name);
        } catch (const std::exception&) {
            // Some devices don't expose FULL_DEVICE_NAME; just print the id.
        }
        text += "\n";
    }
    if (text.empty())
        text = "no OpenVINO devices found\n";
    return text;
}

RIFE::RIFE(const RIFEOptions& o)
    : width_{ o.width }, height_{ o.height } {
    if (o.width < 1 || o.height < 1)
        throw std::runtime_error("invalid clip dimensions");
    if (o.padding < 1)
        throw std::runtime_error("pad must be at least 1");
    if (o.numRequests < 1)
        throw std::runtime_error("threads must be at least 1");

    // ---- read the model (ONNX via the OpenVINO ONNX frontend, or IR .xml) ----
    std::shared_ptr<ov::Model> model;
    try {
        model = readModel(core_, o.modelFile);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to read model '" + o.modelFile + "': " + e.what());
    }

    if (model->inputs().size() != 1)
        throw std::runtime_error("expected a model with exactly one input, got " +
                                 std::to_string(model->inputs().size()) +
                                 " (is this a vs-mlrt RIFE ONNX model?)");
    if (model->outputs().size() != 1)
        throw std::runtime_error("expected a model with exactly one output");

    const auto inShape{ model->input().get_partial_shape() };
    if (inShape.rank().is_dynamic() || inShape.rank().get_length() != 4)
        throw std::runtime_error("expected a 4-D NCHW model input");
    if (inShape[1].is_dynamic())
        throw std::runtime_error("model input channel count must be static");

    inputChannels_ = static_cast<int>(inShape[1].get_length());
    if (inputChannels_ == 11) {
        // v1 layout: the graph does no padding itself, we pad to a multiple of `padding`.
        paddedWidth_ = (width_ + o.padding - 1) / o.padding * o.padding;
        paddedHeight_ = (height_ + o.padding - 1) / o.padding * o.padding;
    } else if (inputChannels_ == 7) {
        // v2 layout: the graph pads internally, feed the raw resolution.
        paddedWidth_ = width_;
        paddedHeight_ = height_;
    } else {
        throw std::runtime_error("unsupported model input with " + std::to_string(inputChannels_) +
                                 " channels (expected 11 for vs-mlrt 'rife/' models or 7 for 'rife_v2/' models)");
    }

    // ---- fix the input shape to the padded clip resolution, batch 1 ----
    try {
        model->reshape({ 1, inputChannels_, paddedHeight_, paddedWidth_ });
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string{ "failed to reshape model to " } +
                                 std::to_string(paddedWidth_) + "x" + std::to_string(paddedHeight_) + ": " + e.what());
    }

    // ---- device properties ----
    ov::AnyMap props;
    if (o.precision == "bf16")
        props[ov::hint::inference_precision.name()] = ov::element::bf16;
    else if (o.precision == "f32")
        props[ov::hint::inference_precision.name()] = ov::element::f32;
    else if (o.precision == "f16")
        props[ov::hint::inference_precision.name()] = ov::element::f16;
    else
        throw std::runtime_error("precision must be \"f32\", \"bf16\" or \"f16\"");

    std::string hint{ o.hint };
    if (hint.empty())
        hint = o.numRequests > 1 ? "throughput" : "latency";
    if (hint == "latency")
        props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    else if (hint == "throughput")
        props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    else if (hint != "none")
        throw std::runtime_error("hint must be \"latency\", \"throughput\" or \"none\"");

    if (o.numRequests > 1)
        props[ov::hint::num_requests.name()] = static_cast<uint32_t>(o.numRequests);
    if (o.ovThreads > 0) {
        props[ov::inference_num_threads.name()] = o.ovThreads;
        // By default the CPU plugin ignores SMT siblings and caps the thread
        // count at the number of physical cores. When the user asks for an
        // explicit thread count, honour it even if it exceeds the physical
        // core count (e.g. a VM that exposes 2 vCPUs on one core).
        props[ov::hint::enable_hyper_threading.name()] = true;
    }
    if (!o.cacheDir.empty())
        core_.set_property(ov::cache_dir(o.cacheDir));

    // ---- compile ----
    try {
        compiled_ = core_.compile_model(model, o.device, props);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to compile model on device '" + o.device + "': " + e.what());
    }

    // ---- request pool ----
    // Constant planes (grid / multipliers) for v1 models are computed once here
    // and copied into every request's input tensor; per frame only channels
    // 0..6 are rewritten.
    const size_t planeSize{ static_cast<size_t>(paddedWidth_) * paddedHeight_ };
    std::vector<float> gridX, gridY;
    if (inputChannels_ == 11) {
        gridX.resize(planeSize);
        gridY.resize(planeSize);
        for (int y = 0; y < paddedHeight_; y++) {
            for (int x = 0; x < paddedWidth_; x++) {
                gridX[static_cast<size_t>(y) * paddedWidth_ + x] = 2.0f * x / (paddedWidth_ - 1) - 1.0f;
                gridY[static_cast<size_t>(y) * paddedWidth_ + x] = 2.0f * y / (paddedHeight_ - 1) - 1.0f;
            }
        }
    }

    try {
        for (int i = 0; i < o.numRequests; i++) {
            auto r{ std::make_unique<Request>() };
            r->request = compiled_.create_infer_request();
            r->input = r->request.get_input_tensor();
            if (r->input.get_element_type() != ov::element::f32)
                throw std::runtime_error("model input must be float32");
            const auto shape{ r->input.get_shape() };
            if (shape.size() != 4 || shape[0] != 1 || shape[1] != static_cast<size_t>(inputChannels_) ||
                shape[2] != static_cast<size_t>(paddedHeight_) || shape[3] != static_cast<size_t>(paddedWidth_))
                throw std::runtime_error("unexpected input tensor shape after reshape");

            auto data{ r->input.data<float>() };
            std::fill_n(data, planeSize * inputChannels_, 0.0f);
            if (inputChannels_ == 11) {
                std::memcpy(data + 7 * planeSize, gridX.data(), sizeof(float) * planeSize);
                std::memcpy(data + 8 * planeSize, gridY.data(), sizeof(float) * planeSize);
                fillPlane(data + 9 * planeSize, planeSize, 2.0f / (paddedWidth_ - 1));
                fillPlane(data + 10 * planeSize, planeSize, 2.0f / (paddedHeight_ - 1));
            }
            freeList_.push_back(r.get());
            pool_.push_back(std::move(r));
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string{ "failed to create inference requests: " } + e.what());
    }

    semaphore_ = std::make_unique<std::counting_semaphore<>>(o.numRequests);
}

RIFE::~RIFE() = default;

RIFE::Request* RIFE::acquire() {
    semaphore_->acquire();
    std::lock_guard<std::mutex> lock{ poolMutex_ };
    Request* r{ freeList_.back() };
    freeList_.pop_back();
    return r;
}

void RIFE::release(Request* r) {
    {
        std::lock_guard<std::mutex> lock{ poolMutex_ };
        freeList_.push_back(r);
    }
    semaphore_->release();
}

void RIFE::fillInput(Request& r, const float* src0R, const float* src0G, const float* src0B,
                     const float* src1R, const float* src1G, const float* src1B,
                     int width, int height, std::ptrdiff_t stride, float timestep) const {
    const size_t planeSize{ static_cast<size_t>(paddedWidth_) * paddedHeight_ };
    auto data{ r.input.data<float>() };

    copyPlanePadded(data + 0 * planeSize, src0R, width, height, stride, paddedWidth_, paddedHeight_);
    copyPlanePadded(data + 1 * planeSize, src0G, width, height, stride, paddedWidth_, paddedHeight_);
    copyPlanePadded(data + 2 * planeSize, src0B, width, height, stride, paddedWidth_, paddedHeight_);
    copyPlanePadded(data + 3 * planeSize, src1R, width, height, stride, paddedWidth_, paddedHeight_);
    copyPlanePadded(data + 4 * planeSize, src1G, width, height, stride, paddedWidth_, paddedHeight_);
    copyPlanePadded(data + 5 * planeSize, src1B, width, height, stride, paddedWidth_, paddedHeight_);
    fillPlane(data + 6 * planeSize, planeSize, timestep);
    // channels 7..10 (v1 only) were filled once at construction
}

void RIFE::process(const float* src0R, const float* src0G, const float* src0B,
                   const float* src1R, const float* src1G, const float* src1B,
                   float* dstR, float* dstG, float* dstB,
                   int width, int height, std::ptrdiff_t stride, float timestep) {
    if (width != width_ || height != height_)
        throw std::runtime_error("frame dimensions do not match the compiled model");

    Request* r{ acquire() };
    try {
        fillInput(*r, src0R, src0G, src0B, src1R, src1G, src1B, width, height, stride, timestep);
        r->request.infer();

        ov::Tensor out{ r->request.get_output_tensor() };
        const auto shape{ out.get_shape() };
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
            shape[2] < static_cast<size_t>(height) || shape[3] < static_cast<size_t>(width))
            throw std::runtime_error("unexpected output tensor shape");
        if (out.get_element_type() != ov::element::f32)
            throw std::runtime_error("model output must be float32");

        const size_t outW{ shape[3] };
        const size_t outPlane{ shape[2] * shape[3] };
        const float* o{ out.data<const float>() };
        float* dst[3]{ dstR, dstG, dstB };
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < height; y++)
                std::memcpy(dst[c] + y * stride, o + c * outPlane + static_cast<size_t>(y) * outW, sizeof(float) * width);
        }
    } catch (...) {
        release(r);
        throw;
    }
    release(r);
}

} // namespace rifeov
