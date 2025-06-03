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
             InputSize input_size = InputSize::kInputSize512)
      : inference_device_(inference_device), input_size_(input_size)
  {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
    session_options_ = Ort::SessionOptions();
    session_options_.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    switch (inference_device_)
    {
      case InferenceDevice::kCPU:
        break;
      case InferenceDevice::kCUDA:
      {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;  // Use the first CUDA device
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchDefault;
        cuda_options.gpu_mem_limit = 0;  // Use all available GPU memory
        cuda_options.arena_extend_strategy = 1;
        cuda_options.do_copy_in_default_stream = 1;
        cuda_options.has_user_compute_stream = 0;
        cuda_options.default_memory_arena_cfg = nullptr;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      }
      break;
      default:
        throw std::runtime_error("Unsupported inference device");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    session_ = std::make_unique<Ort::Session>(env_, model_path.string().c_str(), session_options_);

    configureNodes();
  }

  ~SuperPoint() = default;
  void setInputSize(InputSize input_size) { input_size_ = input_size; }

  // Method to process an image and extract keypoints and descriptors
  Result extract(const cv::Mat& image)
  {
    double scale = 1.0;
    cv::Mat preprocessed = preprocessImage(
        image, [](size_t a, size_t b) { return std::max(a, b); }, scale, cv::INTER_AREA);

    auto output_tensor = inference(preprocessed);
    return postprocess(output_tensor, scale);
  }

private:
  void configureNodes()
  {
    if (!session_)
    {
      throw std::runtime_error("Session is not initialized.");
    }

    auto num_input_nodes = session_->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; ++i)
    {
      Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
      input_node_names_.push_back(input_name.release());
      input_node_shapes_.push_back(session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    auto num_output_nodes = session_->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; ++i)
    {
      Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
      output_node_names_.push_back(output_name.release());
      output_node_shapes_.push_back(session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
  }

  cv::Mat preprocessImage(const cv::Mat& image,
                          std::function<size_t(size_t, size_t)> fn,
                          double& scale,
                          int interp_method = cv::INTER_AREA) const
  {
    // resize the image based on input size and convert to grayscale
    cv::Mat normalize_image, resized_image;
    cv::Size img_size = image.size();
    scale = static_cast<double>(static_cast<uint32_t>(input_size_))
            / static_cast<double>(fn(img_size.width, img_size.height));
    cv::Size new_size(std::lround(img_size.width * scale), std::lround(img_size.height * scale));
    cv::resize(image, resized_image, new_size, 0, 0, interp_method);

    if (resized_image.channels() == 3)
    {
      cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2GRAY);
    }
    else if (resized_image.channels() != 1)
    {
      throw std::runtime_error("Unsupported image format: " + std::to_string(resized_image.channels()));
    }

    // normalize the image to [0, 1] range
    resized_image.convertTo(normalize_image, CV_32F, 1.0 / 255.0);

    return normalize_image;
  }

  std::vector<Ort::Value> inference(const cv::Mat& preprocessed_image)
  {
    input_node_shapes_[0] = {1, 1, preprocessed_image.rows, preprocessed_image.cols};
    const size_t src_input_tensor_size =
        std::accumulate(input_node_shapes_[0].begin(), input_node_shapes_[0].end(), 1, std::multiplies<size_t>());

    std::vector<float> src_input_tensor_values(preprocessed_image.begin<float>(), preprocessed_image.end<float>());

    std::vector<Ort::Value> input_tensors;

    // TODO(deankh): if CUDA is used, allocate memory on GPU
    //               see https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    input_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info, src_input_tensor_values.data(),
                                                            src_input_tensor_size, input_node_shapes_[0].data(),
                                                            input_node_shapes_[0].size()));

    try
    {
      if (!session_)
      {
        throw std::runtime_error("Session is not initialized.");
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      auto output_tensor = session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(),
                                         input_tensors.size(), output_node_names_.data(), output_node_names_.size());
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> duration = end_time - start_time;
      std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
      //  check output format
      for (const auto& tensor : output_tensor)
      {
        if (!tensor.IsTensor() || !tensor.HasValue())
        {
          throw std::runtime_error("Output tensor is not valid.");
        }
      }

      return output_tensor;
    }
    catch (const std::exception& e)
    {
      std::cerr << "Ort::Exception: " << e.what() << std::endl;
      throw;
    }

    return std::vector<Ort::Value>{};  // Placeholder for actual inference logic
  }

  Result postprocess(const std::vector<Ort::Value>& tensor, double scale)
  {
    const std::vector<int64_t> keypoints_shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    Result result;
    std::vector<cv::Point2f>& keypoints = result.keypoints;
    keypoints.reserve(keypoints_shape[1]);
    const int64_t* const keypoints_data = tensor[0].GetTensorData<int64_t>();

    for (size_t i = 0; i < keypoints_shape[1] * 2; i += 2)
    {
      keypoints.emplace_back(static_cast<float>(keypoints_data[i]) / scale,
                             static_cast<float>(keypoints_data[i + 1]) / scale);
    }
    const std::vector<int64_t> score_shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<float>& scores = result.scores;
    scores.reserve(keypoints.size());
    const float* const scores_data = tensor[1].GetTensorData<float>();
    for (size_t i = 0; i < score_shape[1]; ++i)
    {
      scores.push_back(scores_data[i]);
    }

    // 1 x Num x 256
    const std::vector<int64_t> descriptors_shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
    // std::cout << "Descriptors shape: " << descriptors_shape[0] << "x" << descriptors_shape[1] << "x"
    //           << descriptors_shape[2] << std::endl;
    const float* const descriptors_data = (const float* const)tensor[2].GetTensorData<float>();
    result.descriptors = cv::Mat(descriptors_shape[1], descriptors_shape[2], CV_32F, (void*)descriptors_data);

    return result;
  }

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