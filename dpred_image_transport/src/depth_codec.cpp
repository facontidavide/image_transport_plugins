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

#include "depth_codec.hpp"

#include <zstd.h>

// Runtime AVX2 dispatch for the dictionary probe: released binaries build
// with portable flags, so the SIMD path is selected per-CPU at load time
// (GCC/Clang on x86-64; other targets use the scalar probe).
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define DEPTH_CODEC_AVX2_DISPATCH 1
#include <immintrin.h>
#endif

#include <algorithm>
#include <bit>  // NOLINT(build/include_order) -- cpplint predates C++20 headers
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace depth_codec
{
namespace
{

[[noreturn]] void fail(const char * what)
{
  throw std::runtime_error(what);
}

// ---- zstd stage with per-thread context reuse ------------------------------
struct ZstdCtx
{
  ZSTD_CCtx * c = ZSTD_createCCtx();
  ZSTD_DCtx * d = ZSTD_createDCtx();
  ~ZstdCtx()
  {
    ZSTD_freeCCtx(c);
    ZSTD_freeDCtx(d);
  }
};

ZstdCtx & tl_zstd()
{
  static thread_local ZstdCtx ctx;
  return ctx;
}

// Compress src and append the zstd frame to `out` (in place: no
// intermediate buffer, existing capacity is reused across calls).
void zstd_append(std::vector<uint8_t> & out, const uint8_t * src, size_t n, int level)
{
  const size_t base = out.size();
  const size_t bound = ZSTD_compressBound(n);
  out.resize(base + bound);
  const size_t k = ZSTD_compressCCtx(tl_zstd().c, out.data() + base, bound, src, n, level);
  if (ZSTD_isError(k)) {
    fail(ZSTD_getErrorName(k));
  }
  out.resize(base + k);
}

std::vector<uint8_t> zstd_unpack(const uint8_t * comp, size_t comp_size)
{
  const unsigned long long sz = ZSTD_getFrameContentSize(comp, comp_size);  // NOLINT
  if (sz == ZSTD_CONTENTSIZE_ERROR || sz == ZSTD_CONTENTSIZE_UNKNOWN) {
    fail("bad zstd frame");
  }
  std::vector<uint8_t> out(static_cast<size_t>(sz));
  const size_t k = ZSTD_decompressDCtx(tl_zstd().d, out.data(), out.size(), comp, comp_size);
  if (ZSTD_isError(k)) {
    fail(ZSTD_getErrorName(k));
  }
  if (k != out.size()) {
    fail("zstd_unpack: size mismatch");
  }
  return out;
}

// ---- little-endian helpers --------------------------------------------------
inline void put_u32(std::vector<uint8_t> & v, uint32_t x)
{
  for (int i = 0; i < 4; ++i) {
    v.push_back(static_cast<uint8_t>(x >> (8 * i)));
  }
}

inline uint32_t get_u32(const uint8_t * p)
{
  uint32_t x = 0;
  for (int i = 0; i < 4; ++i) {
    x |= static_cast<uint32_t>(p[i]) << (8 * i);
  }
  return x;
}

// Total-order-preserving bijection: unsigned comparison of the mapped word
// equals IEEE-754 total order of the float (negatives fixed up, NaN at the
// top).
inline uint32_t float_to_ord(uint32_t w)
{
  return (w >> 31) ? ~w : (w | 0x80000000u);
}

// JPEG-LS / LOCO-I median-edge-detector predictor, branchless clamp form.
inline int32_t med_predict(int32_t a, int32_t b, int32_t c)
{
  const int32_t mn = std::min(a, b);
  const int32_t mx = std::max(a, b);
  return std::clamp(a + b - c, mn, mx);
}

// ---- shared prediction core --------------------------------------------------
// MED-predict each uint16 in raster order from its left/up/up-left
// neighbours, zigzag the mod-2^16 residual (bijective for any jump size)
// and split it into a low-byte and a high-byte plane. For smooth depth the
// planes are mostly zeros, which the zstd stage then collapses.
void predict_pack(const uint16_t * vals, uint32_t w, uint32_t h, uint8_t * lo, uint8_t * hi)
{
  const auto emit = [&](size_t i, int32_t pred) {
      const int16_t s = static_cast<int16_t>(static_cast<uint16_t>(vals[i] - pred));
      const uint16_t z = static_cast<uint16_t>((s << 1) ^ (s >> 15));
      lo[i] = static_cast<uint8_t>(z);
      hi[i] = static_cast<uint8_t>(z >> 8);
    };
  if (w && h) {
    emit(0, 0);
  }
  for (uint32_t x = 1; x < w; ++x) {
    emit(x, vals[x - 1]);
  }
  for (uint32_t y = 1; y < h; ++y) {
    const size_t row = static_cast<size_t>(y) * w;
    emit(row, vals[row - w]);
    const uint16_t * up = vals + row - w;
    const uint16_t * cur = vals + row;
    for (uint32_t x = 1; x < w; ++x) {
      emit(row + x, med_predict(cur[x - 1], up[x], up[x - 1]));
    }
  }
}

// Inverse of predict_pack. Reconstruction is serial along a row (each
// prediction needs the value just decoded), but row y+1 at column c only
// needs row y up to column c: processing R rows along a skewed diagonal
// ("wavefront") therefore runs R independent dependency chains that the
// out-of-order core overlaps. Pure decoder-side optimization: the format
// and the results are identical to the serial scan.
void predict_unpack(const uint8_t * lo, const uint8_t * hi, uint32_t w, uint32_t h, uint16_t * vals)
{
  const auto unstep = [&](size_t i, int32_t pred) -> uint16_t {
      const uint16_t z = static_cast<uint16_t>(lo[i] | (hi[i] << 8));
      const uint16_t r = static_cast<uint16_t>((z >> 1) ^ (~(z & 1) + 1));
      const uint16_t k = static_cast<uint16_t>(pred + r);
      vals[i] = k;
      return k;
    };
  if (w == 0 || h == 0) {
    return;
  }
  // First row: pure left-prediction chain.
  unstep(0, 0);
  for (uint32_t x = 1; x < w; ++x) {
    unstep(x, vals[x - 1]);
  }

  constexpr uint32_t R = 4;  // interleaved rows = parallel dependency chains
  uint32_t y = 1;
  if (w >= 2 * R) {
    for (; y + R <= h; y += R) {
      uint16_t left[R] = {};
      // Ramp-up: row y+r starts one diagonal step after row y+r-1, which
      // keeps the in-strip dependency satisfied (row r reads row r-1 one
      // step behind).
      for (uint32_t t = 0; t < R; ++t) {
        for (uint32_t r = 0; r <= t; ++r) {
          const uint32_t c = t - r;
          const size_t row = static_cast<size_t>(y + r) * w;
          const uint16_t * up = vals + row - w;
          left[r] = (c == 0) ?
            unstep(row, up[0]) :
            unstep(row + c, med_predict(left[r], up[c], up[c - 1]));
        }
      }
      // Steady state: all R chains active, no bounds checks.
      for (uint32_t t = R; t < w; ++t) {
        for (uint32_t r = 0; r < R; ++r) {
          const uint32_t c = t - r;
          const size_t row = static_cast<size_t>(y + r) * w;
          const uint16_t * up = vals + row - w;
          left[r] = unstep(row + c, med_predict(left[r], up[c], up[c - 1]));
        }
      }
      // Drain: finish the trailing columns of the lower rows.
      for (uint32_t t = w; t < w + R - 1; ++t) {
        for (uint32_t r = t - w + 1; r < R; ++r) {
          const uint32_t c = t - r;
          const size_t row = static_cast<size_t>(y + r) * w;
          const uint16_t * up = vals + row - w;
          left[r] = unstep(row + c, med_predict(left[r], up[c], up[c - 1]));
        }
      }
    }
  }
  // Remaining rows (strip remainder, or narrow images): serial scan.
  for (; y < h; ++y) {
    const size_t row = static_cast<size_t>(y) * w;
    int32_t left = unstep(row, vals[row - w]);
    const uint16_t * up = vals + row - w;
    for (uint32_t x = 1; x < w; ++x) {
      left = unstep(row + x, med_predict(left, up[x], up[x - 1]));
    }
  }
}

// ---- per-image value dictionary, bucketized + SIMD-probed -------------------
// Maps each 32-bit pattern to a first-seen id. 8 keys per cache-line bucket,
// all compared at once; a per-bucket count masks stale slots. The found
// branch is ~99% predictable (only first occurrences miss), which keeps the
// pipeline from flushing. Returns false if > 65536 distinct patterns.
struct alignas(64) DictBucket
{
  uint32_t keys[8];
  uint16_t vals[8];
  uint16_t cnt;
  uint16_t pad[7];
};
static_assert(sizeof(DictBucket) == 64, "bucket must be one cache line");

bool build_value_dict_scalar(
  const uint32_t * words, size_t n,
  std::vector<uint32_t> & entries, std::vector<uint16_t> & idx)
{
  constexpr size_t nb = 1u << 14;  // 16384 buckets x 8 slots, > 2x max load
  constexpr size_t max_dict = 1u << 16;
  std::vector<DictBucket> table(nb);  // zero-init: all counts start at 0
  entries.clear();
  entries.reserve(1u << 12);
  idx.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t k32 = words[i];
    size_t b = (k32 * 2654435761u) >> 18;  // top 14 bits
    uint32_t id;
    for (;; ) {
      DictBucket & bucket = table[b];
      uint32_t m = 0;
      for (unsigned k = 0; k < bucket.cnt; ++k) {
        if (bucket.keys[k] == k32) {
          m = 1u << k;
          break;
        }
      }
      if (m) {
        id = bucket.vals[std::countr_zero(m)];
        break;
      }
      if (bucket.cnt < 8) {
        if (entries.size() >= max_dict) {
          return false;
        }
        id = static_cast<uint32_t>(entries.size());
        bucket.keys[bucket.cnt] = k32;
        bucket.vals[bucket.cnt] = static_cast<uint16_t>(id);
        ++bucket.cnt;
        entries.push_back(k32);
        break;
      }
      b = (b + 1) & (nb - 1);  // bucket full: spill to the next one
    }
    idx[i] = static_cast<uint16_t>(id);
  }
  return true;
}

#if defined(DEPTH_CODEC_AVX2_DISPATCH)
// Verbatim copy of build_value_dict_scalar with the probe replaced by one
// 8-wide SIMD compare. Compiled with the avx2 target attribute so portable
// (non -mavx2) builds still contain it and can select it at runtime; a
// shared inline body cannot carry a per-caller target attribute.
__attribute__((target("avx2"))) bool build_value_dict_avx2(
  const uint32_t * words, size_t n,
  std::vector<uint32_t> & entries, std::vector<uint16_t> & idx)
{
  constexpr size_t nb = 1u << 14;
  constexpr size_t max_dict = 1u << 16;
  std::vector<DictBucket> table(nb);
  entries.clear();
  entries.reserve(1u << 12);
  idx.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t k32 = words[i];
    size_t b = (k32 * 2654435761u) >> 18;
    uint32_t id;
    for (;; ) {
      DictBucket & bucket = table[b];
      const __m256i vk = _mm256_set1_epi32(static_cast<int32_t>(k32));
      const __m256i keys =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bucket.keys));
      uint32_t m = static_cast<uint32_t>(
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(keys, vk))));
      m &= (1u << bucket.cnt) - 1;
      if (m) {
        id = bucket.vals[std::countr_zero(m)];
        break;
      }
      if (bucket.cnt < 8) {
        if (entries.size() >= max_dict) {
          return false;
        }
        id = static_cast<uint32_t>(entries.size());
        bucket.keys[bucket.cnt] = k32;
        bucket.vals[bucket.cnt] = static_cast<uint16_t>(id);
        ++bucket.cnt;
        entries.push_back(k32);
        break;
      }
      b = (b + 1) & (nb - 1);
    }
    idx[i] = static_cast<uint16_t>(id);
  }
  return true;
}
#endif  // DEPTH_CODEC_AVX2_DISPATCH

bool build_value_dict(
  const uint32_t * words, size_t n,
  std::vector<uint32_t> & entries, std::vector<uint16_t> & idx)
{
#if defined(DEPTH_CODEC_AVX2_DISPATCH)
  static const bool use_avx2 = __builtin_cpu_supports("avx2");
  if (use_avx2) {
    return build_value_dict_avx2(words, n, entries, idx);
  }
#endif
  return build_value_dict_scalar(words, n, entries, idx);
}

// ---- dpred payload (32FC1) ---------------------------------------------------
// A single zstd frame whose decompressed content is (little-endian):
//   u8 mode           0 = fallback (raw words), 1 = dictionary
//   u8 idx_bytes      always 2
//   u32 dict_size     ds <= 65536
//   u32 x ds          dictionary, sorted by float total order
//   u8  x n           low bytes of zigzag residuals
//   u8  x n           high bytes of zigzag residuals
void dpred_encode(
  const float * data, uint32_t w, uint32_t h, int level, std::vector<uint8_t> & out)
{
  const size_t n = static_cast<size_t>(w) * h;
  const uint32_t * words = reinterpret_cast<const uint32_t *>(data);

  // Pass 1: dictionary + first-seen indices.
  std::vector<uint32_t> entries;
  std::vector<uint16_t> idx;
  if (!build_value_dict(words, n, entries, idx)) {
    // > 65536 distinct values (e.g. full-precision float depth): store raw.
    std::vector<uint8_t> plain(1 + n * 4);
    plain[0] = 0;
    std::memcpy(plain.data() + 1, words, n * 4);
    zstd_append(out, plain.data(), plain.size(), level);
    return;
  }

  // Sort the dictionary by float total order and remap indices so that
  // index distance ~ depth distance (what makes MED residuals small).
  // LSD radix sort (two 16-bit digits) over key = ord<<16 | original_index.
  const size_t ds = entries.size();
  std::vector<uint64_t> keys_a(ds), keys_b(ds);
  for (size_t k = 0; k < ds; ++k) {
    keys_a[k] = (static_cast<uint64_t>(float_to_ord(entries[k])) << 16) | k;
  }
  {
    std::vector<uint32_t> hist(1u << 16);
    uint64_t * src = keys_a.data();
    uint64_t * dst = keys_b.data();
    for (const int shift : {16, 32}) {
      std::fill(hist.begin(), hist.end(), 0);
      for (size_t k = 0; k < ds; ++k) {
        ++hist[(src[k] >> shift) & 0xFFFF];
      }
      uint32_t sum = 0;
      for (uint32_t & slot : hist) {
        const uint32_t c = slot;
        slot = sum;
        sum += c;
      }
      for (size_t k = 0; k < ds; ++k) {
        dst[hist[(src[k] >> shift) & 0xFFFF]++] = src[k];
      }
      std::swap(src, dst);
    }
  }
  std::vector<uint16_t> rank(ds);
  std::vector<uint32_t> sorted_entries(ds);
  for (size_t k = 0; k < ds; ++k) {
    const uint32_t orig = static_cast<uint32_t>(keys_a[k] & 0xFFFF);
    rank[orig] = static_cast<uint16_t>(k);
    sorted_entries[k] = entries[orig];
  }
  for (size_t i = 0; i < n; ++i) {
    idx[i] = rank[idx[i]];
  }

  std::vector<uint8_t> plain(6 + ds * 4 + n * 2);
  plain[0] = 1;
  plain[1] = 2;  // residuals are always 2 bytes (split into two planes)
  for (int b = 0; b < 4; ++b) {
    plain[2 + b] = static_cast<uint8_t>(ds >> (8 * b));
  }
  static_assert(std::endian::native == std::endian::little, "format is little-endian");
  std::memcpy(plain.data() + 6, sorted_entries.data(), ds * 4);

  uint8_t * lo = plain.data() + 6 + ds * 4;
  predict_pack(idx.data(), w, h, lo, lo + n);
  zstd_append(out, plain.data(), plain.size(), level);
}

void dpred_decode(const uint8_t * comp, size_t comp_size, float * out, uint32_t w, uint32_t h)
{
  const size_t n = static_cast<size_t>(w) * h;
  std::vector<uint8_t> plain = zstd_unpack(comp, comp_size);
  if (plain.empty()) {
    if (n == 0) {
      return;
    }
    fail("dpred_decode: empty payload");
  }
  uint32_t * words = reinterpret_cast<uint32_t *>(out);
  if (plain[0] == 0) {
    if (plain.size() != 1 + n * 4) {
      fail("dpred_decode: size mismatch");
    }
    std::memcpy(words, plain.data() + 1, n * 4);
    return;
  }
  if (plain.size() < 6) {
    fail("dpred_decode: truncated header");
  }
  const size_t ds = get_u32(&plain[2]);
  if (plain.size() != 6 + ds * 4 + n * 2) {
    fail("dpred_decode: size mismatch");
  }
  static_assert(std::endian::native == std::endian::little, "format is little-endian");
  std::vector<uint32_t> dict(ds);
  std::memcpy(dict.data(), plain.data() + 6, ds * 4);
  const uint8_t * lo = plain.data() + 6 + ds * 4;

  std::vector<uint16_t> idx(n);
  predict_unpack(lo, lo + n, w, h, idx.data());
  for (size_t i = 0; i < n; ++i) {
    if (idx[i] >= ds) {
      fail("dpred_decode: bad index");
    }
    words[i] = dict[idx[i]];
  }
}

// ---- public self-describing blob -------------------------------------------
// Layout: 'D' 'P' 'C' '1' | u8 name_len | name | i32 level | u32 w | u32 h
//         | payload. Identical to the standalone depth_codec library.
constexpr char kMethod32[] = "dpred";
constexpr char kMethod16[] = "dpred16";

// Reset `out` (keeping its capacity) and write the blob header; the payload
// is then appended in place by the caller.
void begin_blob(
  std::vector<uint8_t> & out, const char * name, int level, uint32_t width, uint32_t height)
{
  out.clear();
  const char magic[4] = {'D', 'P', 'C', '1'};
  out.insert(out.end(), magic, magic + 4);
  const std::string name_str = name;
  out.push_back(static_cast<uint8_t>(name_str.size()));
  out.insert(out.end(), name_str.begin(), name_str.end());
  put_u32(out, static_cast<uint32_t>(level));
  put_u32(out, width);
  put_u32(out, height);
}

struct ParsedBlob
{
  BlobHeader header;
  const uint8_t * payload;
  size_t payload_size;
};

ParsedBlob parse_blob(const uint8_t * blob, size_t size)
{
  if (size < 5 || std::memcmp(blob, "DPC1", 4) != 0) {
    fail("depth_codec: bad magic");
  }
  size_t pos = 4;
  const uint8_t name_len = blob[pos++];
  if (pos + name_len + 12 > size) {
    fail("depth_codec: truncated header");
  }
  const std::string method(reinterpret_cast<const char *>(blob + pos), name_len);
  pos += name_len;
  ParsedBlob parsed;
  if (method == kMethod32) {
    parsed.header.format = PixelFormat::FLOAT32;
  } else if (method == kMethod16) {
    parsed.header.format = PixelFormat::UINT16;
  } else {
    fail("depth_codec: unknown method (blob from a newer library?)");
  }
  pos += 4;  // level: not needed to decode
  parsed.header.width = get_u32(blob + pos);
  parsed.header.height = get_u32(blob + pos + 4);
  pos += 8;
  parsed.payload = blob + pos;
  parsed.payload_size = size - pos;
  return parsed;
}

}  // namespace

void encode_depth(
  const float * data, uint32_t width, uint32_t height,
  std::vector<uint8_t> & out, int zstd_level)
{
  const int level = std::clamp(zstd_level, 1, 3);
  begin_blob(out, kMethod32, level, width, height);
  dpred_encode(data, width, height, level, out);
}

// 16UC1 payload: a zstd frame of [low plane | high plane] (2 * w * h bytes).
// The pixel value is already a small monotone integer -- its own sorted
// dictionary index -- so no dictionary (and no overflow fallback) is needed.
void encode_depth16(
  const uint16_t * data, uint32_t width, uint32_t height,
  std::vector<uint8_t> & out, int zstd_level)
{
  const int level = std::clamp(zstd_level, 1, 3);
  const size_t n = static_cast<size_t>(width) * height;
  std::vector<uint8_t> plain(n * 2);
  predict_pack(data, width, height, plain.data(), plain.data() + n);
  begin_blob(out, kMethod16, level, width, height);
  zstd_append(out, plain.data(), plain.size(), level);
}

BlobHeader read_header(const uint8_t * blob, size_t size)
{
  return parse_blob(blob, size).header;
}

void decode_depth(const uint8_t * blob, size_t size, float * out)
{
  const ParsedBlob parsed = parse_blob(blob, size);
  if (parsed.header.format != PixelFormat::FLOAT32) {
    fail("decode_depth: blob is not 32FC1");
  }
  dpred_decode(
    parsed.payload, parsed.payload_size, out, parsed.header.width, parsed.header.height);
}

void decode_depth16(const uint8_t * blob, size_t size, uint16_t * out)
{
  const ParsedBlob parsed = parse_blob(blob, size);
  if (parsed.header.format != PixelFormat::UINT16) {
    fail("decode_depth16: blob is not 16UC1");
  }
  const size_t n = static_cast<size_t>(parsed.header.width) * parsed.header.height;
  const std::vector<uint8_t> plain = zstd_unpack(parsed.payload, parsed.payload_size);
  if (plain.size() != n * 2) {
    fail("decode_depth16: size mismatch");
  }
  predict_unpack(
    plain.data(), plain.data() + n, parsed.header.width, parsed.header.height, out);
}

}  // namespace depth_codec
