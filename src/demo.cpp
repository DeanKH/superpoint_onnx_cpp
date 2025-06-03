#include "superpoint.hpp"
#include <eigen3/Eigen/Dense>
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

Eigen::VectorXf computeRowL2Norms(const Eigen::MatrixXf& mat)
{
  return mat.rowwise().norm();
}

// --- cv::Mat -> Eigen::MatrixXf 変換 ---
Eigen::MatrixXf cvMatToEigen(const cv::Mat& mat)
{
  CV_Assert(mat.type() == CV_32F);  // float型前提
  Eigen::MatrixXf eigen_mat(mat.rows, mat.cols);
  for (int i = 0; i < mat.rows; ++i)
    for (int j = 0; j < mat.cols; ++j)
      eigen_mat(i, j) = mat.at<float>(i, j);
  return eigen_mat;
}

Eigen::MatrixXf computeSimilarityMatrix(const cv::Mat& cv_descriptors, const cv::Mat& cv_vocab)
{
  // 1. 変換
  Eigen::MatrixXf descriptors = cvMatToEigen(cv_descriptors);  // NxD
  Eigen::MatrixXf vocab = cvMatToEigen(cv_vocab);              // MxD

  // 2. L2ノルム
  Eigen::VectorXf desc_norms = computeRowL2Norms(descriptors);  // Nx1
  Eigen::VectorXf vocab_norms = computeRowL2Norms(vocab);       // Mx1

  // 3. ノルムの外積 → MxN の正規化係数行列
  Eigen::MatrixXf norm_matrix = vocab_norms * desc_norms.transpose();  // MxN
  std::cout << "Norm matrix shape: " << norm_matrix.rows() << "x" << norm_matrix.cols() << std::endl;

  // 4. 内積計算 vocab * descriptors.T → MxN
  Eigen::MatrixXf dot_product = vocab * descriptors.transpose();  // MxN
  std::cout << "Dot product shape: " << dot_product.rows() << "x" << dot_product.cols() << std::endl;

  // 5. コサイン類似度 = 内積 / ノルム
  Eigen::MatrixXf similarity_matrix = dot_product.array() / norm_matrix.array();  // MxN

  // 6. 転置（元コードと合わせる）
  similarity_matrix.transposeInPlace();  // NxM
  std::cout << "Similarity matrix shape: " << similarity_matrix.rows() << "x" << similarity_matrix.cols() << std::endl;
  return similarity_matrix;
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

  auto start_time = std::chrono::high_resolution_clock::now();

  Eigen::MatrixXf similarity_matrix = computeSimilarityMatrix(result.descriptors, vocab);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end_time - start_time;
  std::cout << "Calc similarity time: " << duration.count() << " ms" << std::endl;

  const double similarity_threshold = 0.6;
  for (size_t i = 0; i < result.keypoints.size(); ++i)
  {
    // similarity_matrixのi行目の要素の内，どれか一つでも閾値を超えるものがあるかどうか
    if (similarity_matrix.row(i).maxCoeff() > similarity_threshold)
    {
      const auto& keypoint = result.keypoints[i];
      cv::circle(image, keypoint, 2, cv::Scalar(0, 255, 0), -1);
    }
  }

  cv::imshow("Keypoints", image);
  cv::waitKey(0);
  return 0;
}
