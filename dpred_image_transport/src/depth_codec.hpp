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

// Vendored "contrib" copy of the dpred codec from
// https://github.com/facontidavide/depth_image_compression
// The blob format is identical, so payloads are interchangeable with the
// standalone library.
//
// dpred losslessly compresses depth images, bit-exact:
//  - 32FC1 (including NaN payloads, +/-inf and -0.0): per-image value
//    dictionary sorted by float total order, 2D MED prediction on the index
//    plane, zigzag residual byte planes, zstd entropy stage.
//  - 16UC1: the pixel value is already a small monotone integer, i.e. its
//    own sorted dictionary index, so the same prediction core applies
//    directly with no dictionary at all.
// See ALGORITHM.md in the source repository.

#ifndef DEPTH_CODEC_HPP_
#define DEPTH_CODEC_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace depth_codec
{

enum class PixelFormat
{
  FLOAT32,  // 32FC1
  UINT16    // 16UC1
};

struct BlobHeader
{
  uint32_t width;
  uint32_t height;
  PixelFormat format;
};

/// Compress width*height float32 (32FC1) depth pixels into `out` (replacing
/// its contents; existing capacity is reused, so a caller that keeps the
/// vector alive across frames pays no steady-state output allocations).
/// zstd_level is clamped to [1, 3]. The blob is self-describing.
void encode_depth(
  const float * data, uint32_t width, uint32_t height,
  std::vector<uint8_t> & out, int zstd_level = 1);

/// Compress width*height uint16 (16UC1) depth pixels into `out`.
void encode_depth16(
  const uint16_t * data, uint32_t width, uint32_t height,
  std::vector<uint8_t> & out, int zstd_level = 1);

/// Parse the self-describing blob header (cheap, no decompression).
/// Use it to size the output buffer and select the decode function.
/// Throws std::runtime_error on malformed or unknown blobs.
BlobHeader read_header(const uint8_t * blob, size_t size);

/// Decode a FLOAT32 blob into a caller-provided buffer of exactly
/// width*height floats (see read_header). Bit-exact round trip; no output
/// allocation or copy. Throws std::runtime_error on malformed input or a
/// pixel-format mismatch.
void decode_depth(const uint8_t * blob, size_t size, float * out);

/// Decode a UINT16 blob into a caller-provided buffer of width*height values.
void decode_depth16(const uint8_t * blob, size_t size, uint16_t * out);

}  // namespace depth_codec

#endif  // DEPTH_CODEC_HPP_
