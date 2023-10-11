#pragma once
#include <cstdint>

// #include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cuda_gaussian_blur(const unsigned char *input,
                                   unsigned char *output, const double *kernel,
                                   int width, int height, int kernel_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row

  if (x < width && y < height) {
    double value = 0.0;
    int offset = kernel_size / 2;

    for (int j = -offset; j <= offset; j++) {
      for (int i = -offset; i <= offset; i++) {
        int x_ = min(max(x + i, 0), width - 1);
        int y_ = min(max(y + j, 0), height - 1);
        value += input[y_ * width + x_] *
                 kernel[(j + offset) * kernel_size + (i + offset)];
      }
    }

    output[y * width + x] = static_cast<unsigned char>(value);
  }
}

// Convert a RGB image to grayscale
// Expects 3 channels, such that the first 3 indices in the input array are the
// RGB values of the first pixel
// ___global__ void rgb_to_grayscale(const uint3 *input,
//                                   std::uint8_t *output, const int width,
//                                   const int height) {
//   const int x = blockIdx.x * blockDim.x + threadIdx.x; // col
//   const int y = blockIdx.y * blockDim.y + threadIdx.y; // row
// }


__global__ void sobel_gradient(const unsigned char *input, std::uint8_t* gradient_x,)
