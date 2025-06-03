#pragma once
#include <cmath>
#include <cstdint>
#include <eigen3/Eigen/Core>
#include <exception>
#include <filesystem>
#include <numeric>
#include <onnxruntime/onnxruntime_cxx_api.h>

#include <onnxruntime_c_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

namespace dklib::experimental
{
class SuperPoint
{
public:
  /** InputSize for SuperPoint input */
  enum class InputSize : uint32_t
  {
    kInputSize512 = 512,
    kInputSize1024 = 1024,
    kInputSize2048 = 2048,
  };

  enum class InferenceDevice : uint32_t
  {
    kCPU = 0,
    kCUDA,
    kNum,
  };

  struct Result
  {
    std::vector<cv::Point2f> keypoints;  // Detected keypoints
    std::vector<float> scores;           // Scores for each keypoint
    cv::Mat descriptors;                 // Descriptors for each keypoint
  };
  SuperPoint(const std::filesystem::path& model_path,
             InferenceDevice inference_device = InferenceDevice::kCPU,
             InputSize input_size = InputSize::kInputSize512);

  ~SuperPoint() = default;

  void setInputSize(InputSize input_size) { input_size_ = input_size; }

  Result extract(const cv::Mat& image)
  {
    double scale = 1.0;
    cv::Mat preprocessed = preprocessImage(
        image, [](size_t a, size_t b) { return std::max(a, b); }, scale, cv::INTER_AREA);

    auto output_tensor = inference(preprocessed);
    return postprocess(output_tensor, scale);
  }

private:
  void configureNodes();

  cv::Mat preprocessImage(const cv::Mat& image,
                          std::function<size_t(size_t, size_t)> fn,
                          double& scale,
                          int interp_method = cv::INTER_AREA) const;

  std::vector<Ort::Value> inference(const cv::Mat& preprocessed_image);

  Result postprocess(const std::vector<Ort::Value>& tensor, double scale);

  InputSize input_size_ = InputSize::kInputSize512;
  InferenceDevice inference_device_ = InferenceDevice::kCPU;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<char*> input_node_names_;
  std::vector<std::vector<int64_t>> input_node_shapes_;
  std::vector<char*> output_node_names_;
  std::vector<std::vector<int64_t>> output_node_shapes_;
};

}  // namespace dklib::experimental