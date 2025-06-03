#include "superpoint.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>

cv::Mat readCSVtoMat(const std::string& filename)
{
  std::ifstream file(filename);
  std::string line;
  std::vector<std::vector<float>> data;

  while (std::getline(file, line))
  {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
      row.push_back(std::stof(cell));
    }

    data.push_back(row);
  }

  // 行数と列数をチェック
  if (data.empty())
  {
    std::cerr << "Empty or invalid CSV file.\n";
    return cv::Mat();
  }

  size_t rows = data.size();
  size_t cols = data[0].size();

  // OpenCVのMatに変換
  cv::Mat mat(rows, cols, CV_32F);
  for (size_t i = 0; i < rows; ++i)
  {
    if (data[i].size() != cols)
    {
      std::cerr << "Inconsistent column size at row " << i << ".\n";
      return cv::Mat();
    }
    for (size_t j = 0; j < cols; ++j)
    {
      mat.at<float>(i, j) = data[i][j];
    }
  }

  return mat;
}

cv::Mat computeRowL2Norms(const cv::Mat& mat)
{
  CV_Assert(mat.type() == CV_32F);  // float型限定

  int rows = mat.rows;
  cv::Mat norms(rows, 1, CV_32F);  // 結果：縦ベクトル（行数 × 1）

  for (int i = 0; i < rows; ++i)
  {
    cv::Mat row = mat.row(i);
    norms.at<float>(i, 0) = std::sqrt(row.dot(row));  // L2ノルム = √(row・row)
  }

  return norms;
}

int main(int argc, char** argv)
{
  std::string model_path = std::string(argv[1]);
  std::string image_path = std::string(argv[2]);
  std::string vocab_path = std::string(argv[3]);
  cv::Mat vocab = readCSVtoMat(vocab_path);
  cv::Mat vocab_norms = computeRowL2Norms(vocab);
  std::cout << "Vocabulary shape: " << vocab.rows << "x" << vocab.cols << std::endl;
  // std::cout << vocab << std::endl;

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Error: Could not read the image at " << image_path << std::endl;
    return -1;
  }

  dklib::experimental::SuperPoint superpoint(model_path, dklib::experimental::SuperPoint::InferenceDevice::kCUDA,
                                             dklib::experimental::SuperPoint::InputSize::kInputSize1024);

  superpoint.extract(image);

  auto result = superpoint.extract(image);
  cv::Mat desc_norms = computeRowL2Norms(result.descriptors);
  cv::Mat norm_matrix = vocab_norms * desc_norms.t();
  std::cout << "Norm matrix shape: " << norm_matrix.rows << "x" << norm_matrix.cols << std::endl;

  // compare cosine similarity with vocabulary
  cv::Mat dot_product = vocab * result.descriptors.t();
  std::cout << "Dot product shape: " << dot_product.rows << "x" << dot_product.cols << std::endl;

  cv::Mat similarity_matrix;
  cv::divide(dot_product, norm_matrix, similarity_matrix);
  similarity_matrix = similarity_matrix.t();

  // for (const auto& keypoint : result.keypoints)
  const double similarity_threshold = 0.6;
  for (size_t i = 0; i < result.keypoints.size(); ++i)
  {
    const float* row_ptr = similarity_matrix.ptr<float>(i);
    for (size_t j = 0; j < vocab.cols; ++j)
    {
      if (row_ptr[j] > similarity_threshold)
      {
        const auto& keypoint = result.keypoints[i];
        cv::circle(image, keypoint, 2, cv::Scalar(0, 255, 0), -1);
        break;
      }
    }
  }

  cv::imshow("Keypoints", image);
  cv::waitKey(0);
  return 0;
}
