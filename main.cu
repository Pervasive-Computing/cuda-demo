#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <string>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

  DEBUG_LOG(stderr, "This is a debug message");

  //

  stbi_image_free(data);

  return EXIT_SUCCESS;
}
