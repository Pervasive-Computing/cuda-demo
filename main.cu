#include <cstdarg>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <string>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "common.hpp"
#include "filters.cuh"
// #include "filters.hpp"

constexpr int KERNEL_SIZE = 101;
constexpr double SIGMA = 20;

double *generate_gaussian_kernel(int N) {
  // std::array<std::array<double, N>, N> kernel;
  double *kernel = new double[N * N];
  // auto kernel = SquareMatrix<double, N>{};
  double sum = 0.0;
  int offset = KERNEL_SIZE / 2;
  for (int y = -offset; y <= offset; y++) {
    for (int x = -offset; x <= offset; x++) {
      // double element = (1.0 / (2 * M_PI * SIGMA * SIGMA)) * std::exp(-(x * x
      // + y * y) / (2 * SIGMA * SIGMA)); kernel.set(x + offset, y + offset,
      // element); sum += element;
      int index = (y + offset) * N + (x + offset);
      kernel[index] = (1.0 / (2 * M_PI * SIGMA * SIGMA)) *
                      std::exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
      sum += kernel[index];
    }
  }

  // Normalize the kernel to make the sum of its coefficients equal to 1.
  for (int y = 0; y < KERNEL_SIZE; y++) {
    for (int x = 0; x < KERNEL_SIZE; x++) {
      // kernel.set(x, y, kernel(x, y) / sum);
      kernel[N * y + x] /= sum;
    }
  }

  return kernel;
}

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s <file.png>\n", argv[0]);
    return EXIT_FAILURE;
  }

  auto const filename = std::string(argv[1]);

  int width, height, channels;
  auto const data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (data == nullptr) {
    std::fprintf(stderr, "Error: could not load image %s\n", filename.c_str());
    return EXIT_FAILURE;
  }

  std::printf("Image %s: %dx%d, %d channels\n", filename.c_str(), width, height,
              channels);

  auto kernel = generate_gaussian_kernel(KERNEL_SIZE);
  // flatten
  // double* kernel = new double[KERNEL_SIZE * KERNEL_SIZE];
  // for (int i = 0; i < KERNEL_SIZE; i++) {
  //   for (int j = 0; j < KERNEL_SIZE; j++) {
  //     kernel[i * KERNEL_SIZE + j] = kernel_mat[j][i];
  //   }
  // }

  // CUDA things

  unsigned char *d_output;
  unsigned char *d_input;
  double *d_kernel;

  char *greyscale_data = new char[width * height * 1];
  { // greyscale input image
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        char r = data[y * width * channels + x * channels + 0];
        char g = data[y * width * channels + x * channels + 1];
        char b = data[y * width * channels + x * channels + 2];
        char grey = (r + g + b) / 3.0;
        greyscale_data[y * width + x] = grey;
      }
    }
  }

  // Write greyscale image
  stbi_write_png("greyscale.png", width, height, 1, greyscale_data, width);

  channels = 1;
  int size = width * height * channels * sizeof(unsigned char);
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, size);
  cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(double));

  cudaMemcpy(d_input, greyscale_data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(double),
             cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  // dim3 block(1, 1);
  dim3 grid(width / block.x, height / block.y);

  std::printf("block: .x = %4d, .y = %4d\n", block.x, block.y);
  std::printf("grid:  .x = %4d, .y = %4d, .z = %4d\n", grid.x, grid.y, grid.z);

  cuda_gaussian_blur<<<grid, block>>>(d_input, d_output, d_kernel, width,
                                      height, KERNEL_SIZE);

  unsigned char *output_data = new unsigned char[size];
  cudaMemcpy(output_data, d_output, size, cudaMemcpyDeviceToHost);

  auto output_filename = filename + ".out.png";
  stbi_write_png(output_filename.c_str(), width, height, channels, output_data,
                 width * channels);
  stbi_image_free(data);
  delete[] kernel;
  delete[] output_data;

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_kernel);

  return EXIT_SUCCESS;
}
