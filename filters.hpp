#pragma once

#include "common.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <stack>
#include <cassert>

template<typename T, std::size_t N>
struct SquareMatrix {
    std::array<std::array<T, N>, N> data;


    auto operator()(int x, int y) -> T& {
        return data[y][x];
    }

    auto operator()(int x, int y) const -> const T& {
        return data[y][x];
    }

    auto size() const -> std::size_t {
        return N;
    }

    // auto set(int x, int y, T value) {
    //     data[y][x] = value;
    // }

    // auto operator[](std::pair<int, int> indices) -> T& {
    //     int x = indices.first;
    //     int y = indices.second;
    //     return data[y][x];
    // }

    // auto operator[](std::pair<int, int> indices) const -> const T& {
    //     int x = indices.first;
    //     int y = indices.second;
    //     return data[y][x];
    // }
};

// Now, you can use the SquareMatrix class like this:

// cpp

// SquareMatrix<int, 3> matrix;

// matrix(0, 1) = 42; // Using the original () syntax
// int value = matrix(0, 1); // Using the original () syntax

// matrix[{1, 2}] = 56; // Using the new [] syntax
// int value2 = matrix[{1, 2}]; // Using the new [] syntax

// This way, you can access and modify elements of the matrix using both the original () syntax and the new [] syntax with pairs of indices.



// Gaussian kernel of size 5x5
constexpr int KERNEL_SIZE = 5;
constexpr double SIGMA = 1.0;  // This value can be adjusted based on the amount of blurring desired

const int SOBEL_SIZE = 3;

template <typename T>
using mat3x3 = std::array<std::array<T, 3>, 3>;


template <typename T>
using vec3 = std::array<T, 3>;

// mat3x3<int> sobelX = {
//     {-1, 0, 1},
//     {-2, 0, 2},
//     {-1, 0, 1}
// };

int sobelX[SOBEL_SIZE][SOBEL_SIZE] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int sobelY[SOBEL_SIZE][SOBEL_SIZE] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


template <std::size_t N>
// constexpr SquareMatrix<double, N> generate_gaussian_kernel() {
constexpr std::array<std::array<double, N>, N> generate_gaussian_kernel() {
    std::array<std::array<double, N>, N> kernel;
    // auto kernel = SquareMatrix<double, N>{};
    double sum = 0.0;
    int offset = KERNEL_SIZE / 2;
    for (int y = -offset; y <= offset; y++) {
        for (int x = -offset; x <= offset; x++) {
            // double element = (1.0 / (2 * M_PI * SIGMA * SIGMA)) * std::exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
            // kernel.set(x + offset, y + offset, element);
            // sum += element;
            kernel[y + offset][x + offset] = (1.0 / (2 * M_PI * SIGMA * SIGMA)) * std::exp(-(x * x + y * y) / (2 * SIGMA * SIGMA));
            sum += kernel[y + offset][x + offset];
        }
    }

    // Normalize the kernel to make the sum of its coefficients equal to 1.
    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            // kernel.set(x, y, kernel(x, y) / sum);
            kernel[y][x] /= sum;
        }
    }

    return kernel;
}

// constexpr std::array<std::array<double, KERNEL_SIZE>, KERNEL_SIZE> gaussian_kernel = generate_gaussian_kernel<KERNEL_SIZE>()


Image gaussian_blur(const Image &image) {
    // generateGaussianKernel();
    const std::array<std::array<double, KERNEL_SIZE>, KERNEL_SIZE> gaussian_kernel = generate_gaussian_kernel<KERNEL_SIZE>();

    // pretty print kernel
    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            std::cout << gaussian_kernel[y][x] << " ";
        }
        std::cout << std::endl;
    }


    auto blurred_image = Image{
        .width = image.width,
        .height = image.height,
        .channels = image.channels,
        .data = new unsigned char[image.width * image.height * image.channels]
    };

    int offset = KERNEL_SIZE / 2;
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            double sum[KERNEL_SIZE] = {0};  // Initialize sum for each channel (assuming at most 5 channels)

            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    int ny = y + ky; // Neighbor y
                    int nx = x + kx; // Neighbor x

                    // Check image boundaries
                    if (ny >= 0 && ny < image.height && nx >= 0 && nx < image.width) {
                        
                        for (int c = 0; c < image.channels; c++) {
                            sum[c] += image.data[(ny * image.width + nx) * image.channels + c] * gaussian_kernel[ky + offset][kx + offset];
                        }
                    }
                }
            }

            for (int c = 0; c < image.channels; c++) {
                blurred_image.data[(y * image.width + x) * image.channels + c] = static_cast<unsigned char>(sum[c]);
            }
        }
    }

    return blurred_image;
}

std::pair<Image, Image> sobel_gradient(const Image &image) {
    int offset = SOBEL_SIZE / 2;

    Image gradientX{
        .width = image.width,
        .height = image.height,
        .channels = 1,  // The gradient is typically single-channel (intensity)
        .data = new unsigned char[image.width * image.height]
    };

    Image gradientY{
        .width = image.width,
        .height = image.height,
        .channels = 1,  // The gradient is typically single-channel (intensity)
        .data = new unsigned char[image.width * image.height]
    };

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            double gx = 0.0;
            double gy = 0.0;

            for (int ky = -offset; ky <= offset; ky++) {
                for (int kx = -offset; kx <= offset; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Ensure we're not out of bounds and use the gray intensity for the gradient (in case of multi-channel images)
                    if (ny >= 0 && ny < image.height && nx >= 0 && nx < image.width) {
                        double intensity = static_cast<double>(image.data[(ny * image.width + nx) * image.channels]); // Assuming first channel for grayscale
                        gx += intensity * sobelX[ky + offset][kx + offset];
                        gy += intensity * sobelY[ky + offset][kx + offset];
                    }
                }
            }

            gradientX.data[y * gradientX.width + x] = static_cast<unsigned char>(std::clamp(gx, 0.0, 255.0));
            gradientY.data[y * gradientY.width + x] = static_cast<unsigned char>(std::clamp(gy, 0.0, 255.0));
        }
    }

    return {gradientX, gradientY};
}

Image compute_magnitude(const Image &gradientX, const Image &gradientY) {
    // Assuming both gradientX and gradientY are of the same dimensions
    Image magnitude{
        .width = gradientX.width,
        .height = gradientX.height,
        .channels = 1,  // The magnitude is typically single-channel (intensity)
        .data = new unsigned char[gradientX.width * gradientX.height]
    };

    for (int y = 0; y < magnitude.height; y++) {
        for (int x = 0; x < magnitude.width; x++) {
            int index = y * magnitude.width + x;

            double gx = static_cast<double>(gradientX.data[index]);
            double gy = static_cast<double>(gradientY.data[index]);

            double mag = std::sqrt(gx * gx + gy * gy);
            magnitude.data[index] = static_cast<unsigned char>(std::clamp(mag, 0.0, 255.0));
        }
    }

    return magnitude;
}

Image non_maximum_suppression(const Image &gradientMagnitude, const Image &gradientX, const Image &gradientY) {
    Image suppressed{
        .width = gradientMagnitude.width,
        .height = gradientMagnitude.height,
        .channels = 1,
        .data = new unsigned char[gradientMagnitude.width * gradientMagnitude.height]
    };

    for (int y = 1; y < suppressed.height - 1; y++) {
        for (int x = 1; x < suppressed.width - 1; x++) {
            double gx = static_cast<double>(gradientX.data[y * suppressed.width + x]);
            double gy = static_cast<double>(gradientY.data[y * suppressed.width + x]);
            double direction = std::atan2(gy, gx); // Gradient direction
            direction = direction * (180.0 / M_PI); // Convert to degrees

            // Normalize direction to [0, 180)
            if (direction < 0) direction += 180;

            // Determine neighbors to check based on gradient direction
            int x1, y1, x2, y2;
            if ((direction >= 0 && direction < 22.5) || (direction >= 157.5 && direction < 180)) {
                // Horizontal direction
                x1 = x + 1; y1 = y;
                x2 = x - 1; y2 = y;
            } else if (direction >= 22.5 && direction < 67.5) {
                // Diagonal: top left to bottom right
                x1 = x + 1; y1 = y - 1;
                x2 = x - 1; y2 = y + 1;
            } else if (direction >= 67.5 && direction < 112.5) {
                // Vertical direction
                x1 = x; y1 = y - 1;
                x2 = x; y2 = y + 1;
            } else {
                // Diagonal: top right to bottom left
                x1 = x - 1; y1 = y - 1;
                x2 = x + 1; y2 = y + 1;
            }

            // Check if the current pixel is a local maximum
            if (gradientMagnitude.data[y * suppressed.width + x] >= gradientMagnitude.data[y1 * suppressed.width + x1] &&
                gradientMagnitude.data[y * suppressed.width + x] >= gradientMagnitude.data[y2 * suppressed.width + x2]) {
                suppressed.data[y * suppressed.width + x] = gradientMagnitude.data[y * suppressed.width + x];
            } else {
                suppressed.data[y * suppressed.width + x] = 0;
            }
        }
    }

    return suppressed;
}

struct ThresholdedImages {
    Image strong;
    Image weak;
};

ThresholdedImages double_thresholding(const Image &image, double lowThreshold, double highThreshold) {
    ThresholdedImages result;

    // Create images for strong and weak pixels
    result.strong = {
        .width = image.width,
        .height = image.height,
        .channels = 1,
        .data = new unsigned char[image.width * image.height]
    };
    
    result.weak = {
        .width = image.width,
        .height = image.height,
        .channels = 1,
        .data = new unsigned char[image.width * image.height]
    };

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            int index = y * image.width + x;
            if (image.data[index] >= highThreshold) {
                result.strong.data[index] = 255;
                result.weak.data[index] = 0;
            } else if (image.data[index] >= lowThreshold) {
                result.strong.data[index] = 0;
                result.weak.data[index] = 255;
            } else {
                result.strong.data[index] = 0;
                result.weak.data[index] = 0;
            }
        }
    }

    return result;
}

Image edge_tracing(const ThresholdedImages &thresholded) {
    Image finalEdges = {
        .width = thresholded.strong.width,
        .height = thresholded.strong.height,
        .channels = 1,
        .data = new unsigned char[thresholded.strong.width * thresholded.strong.height]
    };
    DEBUG_LOG(stderr, "edge_tracing");
    std::copy(thresholded.strong.data, thresholded.strong.data + finalEdges.width * finalEdges.height, finalEdges.data);
    DEBUG_LOG(stderr, "we got past it");


    // Using std::function to declare trace so it can call itself
    std::function<void(int, int)> trace;

    // Helper function to trace edges iteratively using a stack
    trace = [&](int start_x, int start_y) {
        std::stack<std::pair<int, int>> stack;
        stack.push({start_x, start_y});

        while (!stack.empty()) {
            // std::cout << "stack size: " << stack.size() << std::endl;
            auto current = stack.top();
            stack.pop();

            int x = current.first;
            int y = current.second;

            // Check for boundary conditions
            if (x < 0 || x >= finalEdges.width || y < 0 || y >= finalEdges.height) {
                continue;
            }

            // Check for weak pixels in 8-neighbourhood
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;  // Skip the center pixel

                    int nx = x + dx;
                    int ny = y + dy;

                    // Check boundary conditions for the neighbor
                    if (nx < 0 || nx >= finalEdges.width || ny < 0 || ny >= finalEdges.height) {
                        continue;
                    }

                    int index = ny * finalEdges.width + nx;
                    assert(index >= 0 && index < finalEdges.width * finalEdges.height);

                    // If it's a weak pixel, make it strong and push to the stack
                    if (thresholded.weak.data[index] == 255) {
                        finalEdges.data[index] = 255;  // Make it strong
                        thresholded.weak.data[index] = 0;  // Mark as processed
                        stack.push({nx, ny});  // Push to the stack for further tracing
                    }
                }
            }
        }
    };

    // Start the edge tracing from strong pixels
    for (int y = 0; y < finalEdges.height; y++) {
        for (int x = 0; x < finalEdges.width; x++) {
            int index = y * finalEdges.width + x;
            if (finalEdges.data[index] == 255) {  // Strong pixel
                trace(x, y);
            }
        }
    }

    return finalEdges;
}

auto canny(const Image &image, const std::string& prefix) -> Image {
    
    // auto output_image = Image{
    //     .width = image.width,
    //     .height = image.height,
    //     .channels = image.channels,
    //     .data = new unsigned char[image.width * image.height * image.channels]
    // };

    // 1. Noise Reduction using Gaussian Blur
    int step = 0;
    Image blurred_image = gaussian_blur(image);  // Assume you have a gaussianBlur function
    step++;
    blurred_image.write(prefix + "-" + std::to_string(step) + "-canny-blurred.png");

    // 2. Compute Gradient and its magnitude
    Image gx, gy;
    std::tie(gx, gy) = sobel_gradient(blurred_image);  // Assume you have a sobelGradient function
    step++;
    gx.write(prefix + "-" + std::to_string(step) + "-canny-sobel-gradient-x.png");
    gy.write(prefix + "-" + std::to_string(step) + "-canny-sobel-gradient-y.png");

    Image gradient_magnitude = compute_magnitude(gx, gy);  // Compute the magnitude
    step++;
    gradient_magnitude.write(prefix + "-" + std::to_string(step) + "-canny-magnitude.png");

    // 3. Non-maximum suppression
    Image suppressed_image = non_maximum_suppression(gradient_magnitude, gx, gy);  // Assume you have a nonMaximumSuppression function
    step++;
    suppressed_image.write(prefix + "-" + std::to_string(step) + "-canny-non-maximum-suppression.png");

    // 4.1. Double thresholding
    double low_threshold = 50;
    double high_threshold = 150;
    ThresholdedImages thresholded_image = double_thresholding(suppressed_image, low_threshold, high_threshold);  // Assume you have a doubleThresholding function
    step++;
    thresholded_image.strong.write(prefix + "-" + std::to_string(step) + "-canny-strong.png");
    thresholded_image.weak.write(prefix + "-" + std::to_string(step) + "-canny-weak.png");
    
    // 4.2. Tracing edges with hysteresis
    Image final_image = edge_tracing(thresholded_image);  // Assume you have an edgeTracing function
    step++;


    delete [] thresholded_image.strong.data;
    delete [] thresholded_image.weak.data;
    delete [] blurred_image.data;
    delete [] gx.data;
    delete [] gy.data;
    delete [] gradient_magnitude.data;
    delete [] suppressed_image.data;
    
    // return final_image;
    return final_image;
}
