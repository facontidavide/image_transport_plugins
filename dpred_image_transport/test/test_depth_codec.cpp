// Copyright (c) 2026, Davide Faconti
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "depth_codec.hpp"

namespace
{

float from_bits(uint32_t u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Encode + read_header + decode into a preallocated buffer, and require a
// BIT-EXACT payload (memcmp over the raw bytes, so NaN payloads, +/-inf and
// -0.0 are checked too).
void expect_roundtrip_32f(const std::vector<float> & img, uint32_t w, uint32_t h)
{
  std::vector<uint8_t> blob;  // reused across levels: capacity-reuse path
  for (int level : {1, 3}) {
    depth_codec::encode_depth(img.data(), w, h, blob, level);
    const depth_codec::BlobHeader header =
      depth_codec::read_header(blob.data(), blob.size());
    ASSERT_EQ(header.width, w);
    ASSERT_EQ(header.height, h);
    ASSERT_EQ(header.format, depth_codec::PixelFormat::FLOAT32);
    std::vector<float> back(img.size());
    depth_codec::decode_depth(blob.data(), blob.size(), back.data());
    if (!img.empty()) {
      ASSERT_EQ(0, std::memcmp(back.data(), img.data(), img.size() * 4));
    }
  }
}

void expect_roundtrip_16u(const std::vector<uint16_t> & img, uint32_t w, uint32_t h)
{
  std::vector<uint8_t> blob;  // reused across levels: capacity-reuse path
  for (int level : {1, 3}) {
    depth_codec::encode_depth16(img.data(), w, h, blob, level);
    const depth_codec::BlobHeader header =
      depth_codec::read_header(blob.data(), blob.size());
    ASSERT_EQ(header.width, w);
    ASSERT_EQ(header.height, h);
    ASSERT_EQ(header.format, depth_codec::PixelFormat::UINT16);
    std::vector<uint16_t> back(img.size());
    depth_codec::decode_depth16(blob.data(), blob.size(), back.data());
    if (!img.empty()) {
      ASSERT_EQ(0, std::memcmp(back.data(), img.data(), img.size() * 2));
    }
  }
}

std::vector<float> make_depth_frame(uint32_t w, uint32_t h)
{
  std::vector<float> img(static_cast<size_t>(w) * h);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const int q = 1 + static_cast<int>((x * 131 + y * 17) % 5000);
      img[static_cast<size_t>(y) * w + x] = 10100.0f / (static_cast<float>(q) + 1009.0f);
    }
  }
  // Canonical-NaN hole, like an invalid region from a depth camera.
  for (uint32_t y = h / 4; y < h / 2; ++y) {
    for (uint32_t x = w / 3; x < 2 * w / 3; ++x) {
      img[static_cast<size_t>(y) * w + x] = from_bits(0x7FC00000u);
    }
  }
  return img;
}

std::vector<uint16_t> make_depth_frame16(uint32_t w, uint32_t h)
{
  std::vector<uint16_t> img(static_cast<size_t>(w) * h);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      img[static_cast<size_t>(y) * w + x] =
        static_cast<uint16_t>(300 + (x * 131 + y * 17) % 5000);
    }
  }
  // Zero hole, the 16UC1 invalid-pixel convention.
  for (uint32_t y = h / 4; y < h / 2; ++y) {
    for (uint32_t x = w / 3; x < 2 * w / 3; ++x) {
      img[static_cast<size_t>(y) * w + x] = 0;
    }
  }
  return img;
}

}  // namespace

TEST(DepthCodec, depth_frame_roundtrip_32f)
{
  expect_roundtrip_32f(make_depth_frame(97, 53), 97, 53);
  expect_roundtrip_32f(make_depth_frame(640, 480), 640, 480);
}

TEST(DepthCodec, depth_frame_roundtrip_16u)
{
  expect_roundtrip_16u(make_depth_frame16(97, 53), 97, 53);
  expect_roundtrip_16u(make_depth_frame16(640, 480), 640, 480);
}

TEST(DepthCodec, special_values_32f)
{
  const uint32_t bits[] = {
    0x00000000u,  // +0.0
    0x80000000u,  // -0.0
    0x7F800000u,  // +inf
    0xFF800000u,  // -inf
    0x7FC00000u,  // canonical quiet NaN
    0x7FC00001u,  // NaN, non-canonical payload
    0x7F800001u,  // signalling NaN
    0xFFC00000u,  // negative NaN
    0xFFFFFFFFu,  // NaN, all ones
    0x00000001u,  // smallest positive denormal
    0x807FFFFFu,  // negative denormal
    0x7F7FFFFFu,  // FLT_MAX
    0xFF7FFFFFu,  // -FLT_MAX
    0x00800000u,  // FLT_MIN
    0xBF800000u,  // -1.0
    0x3F800000u,  // 1.0
  };
  const uint32_t w = 8;
  const uint32_t h = 8;
  std::vector<float> img(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < img.size(); ++i) {
    img[i] = from_bits(bits[i % std::size(bits)]);
  }
  expect_roundtrip_32f(img, w, h);
}

TEST(DepthCodec, special_values_16u)
{
  // Extremes and large jumps (exercises the mod-2^16 residual wrap).
  const uint16_t vals[] = {0, 65535, 1, 65534, 32768, 32767, 0, 65535};
  const uint32_t w = 8;
  const uint32_t h = 8;
  std::vector<uint16_t> img(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < img.size(); ++i) {
    img[i] = vals[i % std::size(vals)];
  }
  expect_roundtrip_16u(img, w, h);
}

TEST(DepthCodec, dictionary_overflow_falls_back)
{
  // > 65536 distinct bit patterns forces the raw-zstd fallback path (32FC1
  // only: 16UC1 cannot overflow by construction).
  std::mt19937 rng(12345);
  const uint32_t w = 512;
  const uint32_t h = 256;
  std::vector<float> img(static_cast<size_t>(w) * h);
  for (auto & f : img) {
    f = from_bits(rng());
  }
  expect_roundtrip_32f(img, w, h);
}

TEST(DepthCodec, degenerate_shapes)
{
  expect_roundtrip_32f(std::vector<float>(64 * 32, 1.25f), 64, 32);  // constant
  expect_roundtrip_32f({0.5f}, 1, 1);
  expect_roundtrip_32f({1.f, 2.f, 3.f, 4.f, 5.f}, 5, 1);
  expect_roundtrip_32f({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f}, 1, 7);
  expect_roundtrip_16u(std::vector<uint16_t>(64 * 32, 1250), 64, 32);
  expect_roundtrip_16u({1234}, 1, 1);
  expect_roundtrip_16u({1, 2, 3, 4, 5}, 5, 1);
  expect_roundtrip_16u({1, 2, 3, 4, 5, 6, 7}, 1, 7);
}

TEST(DepthCodec, malformed_input_throws)
{
  const uint8_t junk[8] = {'X', 'X', 'X', 'X', 0, 0, 0, 0};
  EXPECT_THROW(depth_codec::read_header(junk, sizeof(junk)), std::runtime_error);

  // Truncated valid blob must throw, not crash.
  const float one = 1.0f;
  std::vector<uint8_t> blob;
  depth_codec::encode_depth(&one, 1, 1, blob, 1);
  float out = 0.0f;
  EXPECT_THROW(depth_codec::decode_depth(blob.data(), blob.size() / 2, &out),
    std::runtime_error);
}

TEST(DepthCodec, pixel_format_mismatch_throws)
{
  const float onef = 1.0f;
  const uint16_t oneu = 1;
  std::vector<uint8_t> blob32;
  std::vector<uint8_t> blob16;
  depth_codec::encode_depth(&onef, 1, 1, blob32, 1);
  depth_codec::encode_depth16(&oneu, 1, 1, blob16, 1);

  uint16_t out16 = 0;
  float out32 = 0.0f;
  EXPECT_THROW(depth_codec::decode_depth16(blob32.data(), blob32.size(), &out16),
    std::runtime_error);
  EXPECT_THROW(depth_codec::decode_depth(blob16.data(), blob16.size(), &out32),
    std::runtime_error);
}
