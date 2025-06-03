#include "superpoint.hpp"
#include <iostream>

int main(int argc, char** argv)
{
  std::string model_path = std::string(argv[1]);
  std::string image_path = std::string(argv[2]);
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Error: Could not read the image at " << image_path << std::endl;
    return -1;
  }

  // cv::imshow("Input Image", image);
  // cv::waitKey(0);

  dklib::experimental::SuperPoint superpoint(model_path, dklib::experimental::SuperPoint::InferenceDevice::kCUDA,
                                             dklib::experimental::SuperPoint::InputSize::kInputSize1024);

  superpoint.extract(image);

  auto result = superpoint.extract(image);
  // plot keypoints on image

  for (const auto& keypoint : result.keypoints)
  {
    cv::circle(image, keypoint, 2, cv::Scalar(0, 255, 0), -1);
  }
  cv::imshow("Keypoints", image);
  cv::waitKey(0);
  return 0;
}
