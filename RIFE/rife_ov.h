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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace rifeov {

// Options describing how a RIFE model should be loaded and compiled.
struct RIFEOptions {
    std::string modelFile;      // path to .onnx or OpenVINO IR .xml (UTF-8)
    std::string device{ "CPU" };// OpenVINO device name, e.g. "CPU", "GPU"
    int width{};                // clip width (unpadded)
    int height{};               // clip height (unpadded)
    int padding{ 32 };          // input dimensions are rounded up to a multiple of this
    int numRequests{ 1 };       // number of ov::InferRequest objects in the pool
    int ovThreads{ 0 };         // INFERENCE_NUM_THREADS (0 = let OpenVINO decide)
    std::string precision{ "f32" }; // "f32" or "bf16" -> INFERENCE_PRECISION_HINT
    std::string hint;           // "latency", "throughput" or "" (auto)
    std::string cacheDir;       // ov::cache_dir (empty = disabled)
};

/*
    RIFE v4.x inference on top of the OpenVINO runtime.

    The class owns one ov::CompiledModel (compiled once at construction for the
    padded clip resolution) and a pool of ov::InferRequest objects. process()
    borrows a request from the pool (blocking when all of them are busy), fills
    its input tensor, runs synchronous inference and writes the cropped result
    back into the destination planes. This mirrors the gpu_thread / semaphore
    scheme of the ncnn-Vulkan plugin so that VapourSynth's parallel filter mode
    can keep several inferences in flight.

    Supported model layouts (vs-mlrt RIFE ONNX exports):
      * v1 ("rife/<name>.onnx"):     input [1, 11, H, W]
                                     = img0 RGB, img1 RGB, timestep, grid x,
                                       grid y, multiplier h, multiplier w.
                                     H and W must be multiples of `padding`;
                                     this class pads with edge replication.
      * v2 ("rife_v2/<name>.onnx"):  input [N, 7, H, W] = img0, img1, timestep;
                                     the graph pads internally (best effort).
*/
class RIFE final {
public:
    explicit RIFE(const RIFEOptions& options);
    ~RIFE();

    RIFE(const RIFE&) = delete;
    RIFE& operator=(const RIFE&) = delete;

    // Interpolate a frame at `timestep` (0..1) between src0 and src1.
    // Planes are 32-bit float, `stride` is in floats and shared by all planes.
    void process(const float* src0R, const float* src0G, const float* src0B,
                 const float* src1R, const float* src1G, const float* src1B,
                 float* dstR, float* dstG, float* dstB,
                 int width, int height, std::ptrdiff_t stride, float timestep);

    int inputChannels() const noexcept { return inputChannels_; }
    int paddedWidth() const noexcept { return paddedWidth_; }
    int paddedHeight() const noexcept { return paddedHeight_; }
    bool isV2() const noexcept { return inputChannels_ == 7; }

    // Human readable list of OpenVINO devices ("CPU: AMD Ryzen ...").
    static std::string listDevices();

private:
    struct Request {
        ov::InferRequest request;
        ov::Tensor input;
        ov::Tensor output;
    };

    Request* acquire();
    void release(Request* r);
    void fillInput(Request& r, const float* src0R, const float* src0G, const float* src0B,
                   const float* src1R, const float* src1G, const float* src1B,
                   int width, int height, std::ptrdiff_t stride, float timestep) const;

    int width_{};
    int height_{};
    int paddedWidth_{};
    int paddedHeight_{};
    int inputChannels_{};

    ov::Core core_;
    ov::CompiledModel compiled_;
    std::vector<std::unique_ptr<Request>> pool_;
    std::vector<Request*> freeList_;
    std::mutex poolMutex_;
    std::unique_ptr<std::counting_semaphore<>> semaphore_;
};

} // namespace rifeov
