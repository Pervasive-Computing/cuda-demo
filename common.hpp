#pragma once

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>

namespace escape_codes {
constexpr const char *reset = "\033[0m";
constexpr const char *red = "\033[31m";
constexpr const char *green = "\033[32m";
constexpr const char *yellow = "\033[33m";
constexpr const char *blue = "\033[34m";
constexpr const char *magenta = "\033[35m";
constexpr const char *cyan = "\033[36m";
} // namespace escape_codes

#ifndef NDEBUG
#define DEBUG_LOG(stream, format, ...)                                         \
  /* The do {...} while(0) structure is a common idiom used in macros. */      \
  /* It allows the macro to be used in all contexts that a normal function     \
   * call could be used. */                                                    \
  /* It creates a compound statement in C/C++ that behaves as a single         \
   * statement. */                                                             \
  do {                                                                         \
    bool is_tty = isatty(fileno(stream));                                      \
    if (is_tty) {                                                              \
      fprintf(stream, "%s%s%s:%s%s%s:%d: ", escape_codes::cyan, __FILE__,      \
              escape_codes::reset, escape_codes::yellow, __func__,             \
              escape_codes::reset, __LINE__);                                  \
    } else {                                                                   \
      fprintf(stream, "%s:%s:%d: ", __FILE__, __func__, __LINE__);             \
    }                                                                          \
    fprintf(stream, format, ##__VA_ARGS__);                                    \
    fprintf(stream, "\n");                                                     \
  } while (0)
#else
// do nothing
#define DEBUG_LOG(stream, format, ...)
#endif

struct rgb {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

struct rgba {
  unsigned char r;
  unsigned char g;
  unsigned char b;
  unsigned char a;
};

struct Image {
  int width;
  int height;
  int channels;
  unsigned char *data;

  // ~Image() {
  //   if (data != nullptr) {
  //   delete [] data;
  //   }
  // }

  auto info() const -> std::string {

    auto const size = width * height * channels;
    auto const size_in_mb = size / (1024.0 * 1024.0);
    auto const size_in_kb = size / 1024.0;

    auto const size_str = size_in_mb > 1.0 ? std::to_string(size_in_mb) + " MB"
                                           : std::to_string(size_in_kb) + " KB";

    return std::string("Image: ") + std::to_string(width) + "x" +
           std::to_string(height) + ", " + std::to_string(channels) +
           " channels, " + size_str;
  }

  auto operator()(int x, int y) -> rgba {
    auto const i = (y * width + x) * channels;
    return {.r = data[i + 0],
            .g = data[i + 1],
            .b = data[i + 2],
            .a = static_cast<unsigned char>(channels == 4 ? data[i + 3] : 255)};
  }

  auto operator()(int x, int y) const -> rgba {
    auto const i = (y * width + x) * channels;
    return {.r = data[i + 0],
            .g = data[i + 1],
            .b = data[i + 2],
            .a = static_cast<unsigned char>(channels == 4 ? data[i + 3] : 255)};
  }

  auto set(int x, int y, const rgb c) {
    auto const i = (y * width + x) * channels;
    data[i + 0] = c.r;
    data[i + 1] = c.g;
    data[i + 2] = c.b;
  }

  auto set(int x, int y, const rgba c) {
    auto const i = (y * width + x) * channels;
    data[i + 0] = c.r;
    data[i + 1] = c.g;
    data[i + 2] = c.b;
    if (channels == 4) {
      data[i + 3] = c.a;
    }
  }

  auto write(const std::string &filename) -> void {
    stbi_write_png(filename.c_str(), width, height, channels, data,
                   width * channels);
  }
};
