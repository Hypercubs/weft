#pragma once

// The Metal side of the translated-kernel runtime.
//
// A CPU module links against runtime/weft_runtime.hpp: buffer addresses, atomics,
// a rendezvous for shuffles. An emitted GPU kernel needs the same names with a
// hardware answer behind them, so this is that surface written in Metal. The
// instruction text is then the same for both targets, which is the whole point:
// one decode, two back ends, and the GPU column becomes a comparison of the same
// generated code rather than of a hand-written twin.
//
// Where the hardware changes the shape of an instruction:
//
//   - a shuffle is one instruction (simd_shuffle) instead of publish-yield-read;
//   - a barrier is a real threadgroup barrier instead of `save pc, return`;
//   - threadgroup memory is a `threadgroup` array instead of a shared arena;
//   - a 64-bit atomic does not exist on this host (tools/atom64_probe.mm), so one
//     is built from the 32-bit atomics there are: a lock over the two halves. That
//     lock has no ordering to stand on — metal_types declares `memory_order_relaxed`
//     and nothing else — so a payload store can escape the unlock and a peer can
//     read a torn pair. The warp back end is exact with it (one thread owns all 32
//     lanes, and every access is an atomic from that one thread); the per-lane back
//     end measured wrong at 65,536 lanes in both shapes it was tried, so it refuses
//     those by name and keeps only the shape that needs no ordering at all — a
//     reduction, which is two 32-bit atomics.
//
// Everything here is `static inline` in one translation unit, so the compiler
// inlines the helpers into the kernel and constant-folds the mode switches.
static const char* kWWeftMslPreamble = R"WEFMSL(
#include <metal_stdlib>
using namespace metal;

// The helpers the emitted body names. Same names, same answers: the bit-level
// ones are pure integer work, and memcpy becomes as_type because there is no
// memcpy in a shader.
// `as_type` rather than a union: the language has the bitcast, so there is one
// spelling of a reinterpret rather than two.
template <class T, class U>
static inline T weft_bitcast(U u) {
  return as_type<T>(u);
}
static inline float weft_f32(uint32_t b) { return as_type<float>(b); }
static inline uint32_t weft_bits(float f) { return as_type<uint32_t>(f); }
static inline uint32_t weft_bitrev32(uint32_t v) {
  v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
  v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
  v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
  v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
  return (v >> 16) | (v << 16);
}
static inline uint16_t weft_h16_bits(float x) { return as_type<uint16_t>((_Float16)x); }
static inline float weft_h16(uint32_t b, int i) {
  return (float)as_type<_Float16>((uint16_t)(b >> (16 * i)));
}
static inline uint32_t weft_f16x2(float hi, float lo) {
  return (uint32_t)weft_h16_bits(lo) | ((uint32_t)weft_h16_bits(hi) << 16);
}
static inline uint16_t weft_bf16_bits(float x) {
  uint32_t u = as_type<uint32_t>(x);
  if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline uint32_t weft_bf16x2(float hi, float lo) {
  return (uint32_t)weft_bf16_bits(lo) | ((uint32_t)weft_bf16_bits(hi) << 16);
}
static inline float weft_bf16(uint32_t b, int i) {
  return as_type<float>((uint32_t)(uint16_t)(b >> (16 * i)) << 16);
}
// e4m3 / e5m2 conversion is integer work plus an exact power-of-two scale, so
// float arithmetic here reproduces the CPU's double arithmetic bit for bit:
// dividing by 512 and multiplying by 65536 are exact in both.
static inline uint8_t weft_e4m3_bits(float x) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;
  if ((u & 0x7FFFFFFFu) == 0u) return sign;
  if (!isfinite(x)) return (uint8_t)(sign | 0x7Fu);
  const int E = (int)((u >> 23) & 0xFFu) - 127 + 7;
  if (E >= 16) return (uint8_t)(sign | 0x7Eu);
  if (E <= 0) {
    const float q = rint(fabs(x) * 512.0f);
    if (q >= 8.0f) return (uint8_t)(sign | 0x08u);
    return (uint8_t)(sign | (uint8_t)q);
  }
  const uint32_t mant = u & 0x7FFFFFu;
  uint32_t r = mant >> 20;
  const uint32_t rem = mant & 0xFFFFFu;
  if (rem > 0x80000u || (rem == 0x80000u && (r & 1u))) r++;
  const uint32_t out = ((uint32_t)E << 3) + r;
  return (uint8_t)(sign | (uint8_t)(out >= 0x7Fu ? 0x7Eu : out));
}
static inline uint32_t weft_e4m3x2(float hi, float lo) {
  return (uint32_t)weft_e4m3_bits(lo) | ((uint32_t)weft_e4m3_bits(hi) << 8);
}
static inline float weft_e4m3(uint32_t b, int i) {
  const uint32_t v = (b >> (8 * i)) & 0xFFu;
  const uint32_t mag = v & 7u, E = (v >> 3) & 0xFu;
  const float d = (E == 0) ? (float)mag / 512.0f : ldexp((float)(8u + mag), (int)E - 10);
  return (float)((v & 0x80u) ? -d : d);
}
static inline uint8_t weft_e5m2_bits(float x) {
  const uint32_t u = as_type<uint32_t>(x);
  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;
  if ((u & 0x7FFFFFFFu) == 0u) return sign;
  if (!isfinite(x)) return (uint8_t)(sign | 0x7Fu);
  const int E = (int)((u >> 23) & 0xFFu) - 127 + 15;
  if (E >= 31) return (uint8_t)(sign | 0x7Bu);
  if (E <= 0) {
    const float q = rint(fabs(x) * 65536.0f);
    if (q >= 4.0f) return (uint8_t)(sign | 0x04u);
    return (uint8_t)(sign | (uint8_t)q);
  }
  const uint32_t mant = u & 0x7FFFFFu;
  uint32_t r = mant >> 21;
  const uint32_t rem = mant & 0x1FFFFFu;
  if (rem > 0x100000u || (rem == 0x100000u && (r & 1u))) r++;
  const uint32_t out = ((uint32_t)E << 2) + r;
  return (uint8_t)(sign | (uint8_t)(out >= 0x7Cu ? 0x7Bu : out));
}
static inline uint32_t weft_e5m2x2(float hi, float lo) {
  return (uint32_t)weft_e5m2_bits(lo) | ((uint32_t)weft_e5m2_bits(hi) << 8);
}
static inline float weft_e5m2(uint32_t b, int i) {
  const uint32_t v = (b >> (8 * i)) & 0xFFu;
  const uint32_t mag = v & 3u, E = (v >> 2) & 0x1Fu;
  if (E == 31u) return (mag == 0u) ? ((v & 0x80u) ? -INFINITY : INFINITY) : NAN;
  const float d = (E == 0) ? (float)mag / 65536.0f : ldexp((float)(4u + mag), (int)E - 17);
  return (float)((v & 0x80u) ? -d : d);
}
static inline float weft_tf32(float x) {
  uint32_t u = as_type<uint32_t>(x);
  if ((u & 0x7F800000u) == 0x7F800000u) return x;
  u += 0x1000u + ((u >> 13) & 1u);
  u &= 0xFFFFE000u;
  return as_type<float>(u);
}
static inline uint32_t weft_shf_l(uint32_t a, uint32_t b, uint32_t c, int clamp) {
  if (clamp) { if (c >= 32u) return a; } else c &= 31u;
  return c ? ((b << c) | (a >> (32u - c))) : b;
}
static inline uint32_t weft_shf_r(uint32_t a, uint32_t b, uint32_t c, int clamp) {
  if (clamp) { if (c >= 32u) return b; } else c &= 31u;
  return c ? ((a >> c) | (b << (32u - c))) : a;
}
static inline uint32_t weft_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t out = 0;
  for (uint32_t i = 0; i < 8u; i++) {
    const uint32_t m = ((i & 4u) ? a : ~a) & ((i & 2u) ? b : ~b) & ((i & 1u) ? c : ~c);
    if (lut & (1u << i)) out |= m;
  }
  return out;
}
static inline uint32_t weft_prmt(uint32_t a, uint32_t b, uint32_t c) {
  const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);
  uint32_t out = 0;
  for (int i = 0; i < 4; i++) {
    const uint32_t sel = (c >> (4 * i)) & 0xFu;
    const uint32_t byte = (uint32_t)((src >> ((sel & 7u) * 8u)) & 0xFFu);
    out |= ((sel & 8u) ? ((byte & 0x80u) ? 0xFFu : 0u) : byte) << (8 * i);
  }
  return out;
}
// mul.hi.u64 / mul.hi.s64: the high 64 bits of a 64x64 product, built from four
// 32x32 multiplies. The scalar `__int128` in the language would express it in one
// line, but this host's back end gives up on a 128-bit multiply in a kernel of
// real size: the library compiles and the *pipeline* then fails with
// XPC_ERROR_CONNECTION_INTERRUPTED, which is what kept kernels/cuda/bits.ptx off
// the GPU column. Four products and a carry chain cost a few instructions and
// compile everywhere.
static inline uint64_t weft_mulhi_u64(uint64_t a, uint64_t b) {
  const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;
  const uint64_t p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;
  const uint64_t cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu)) >> 32;
  return p3 + (p1 >> 32) + (p2 >> 32) + cy;
}
// The signed word is the unsigned product with the two sign extensions taken
// back out: a negative `a` contributed `b` too many times, and vice versa.
static inline int64_t weft_mulhi_s64(int64_t a, int64_t b) {
  uint64_t h = weft_mulhi_u64((uint64_t)a, (uint64_t)b);
  if (a < 0) h -= (uint64_t)b;
  if (b < 0) h -= (uint64_t)a;
  return (int64_t)h;
}

// The packed vector types the SPIR-V front end uses: a <3 x float> in memory is
// 12 bytes, and the state struct has to agree or a load reads the wrong data.
struct weft_f2 { float x, y; };
struct weft_f3 { float x, y, z; };
struct weft_f4 { float x, y, z, w; };
struct weft_i2 { int32_t x, y; };
struct weft_i3 { int32_t x, y, z; };
struct weft_i4 { int32_t x, y, z, w; };
struct weft_u2 { uint32_t x, y; };
struct weft_u3 { uint32_t x, y, z; };
struct weft_u4 { uint32_t x, y, z, w; };
struct weft_l2 { uint64_t x, y; };
struct weft_l3 { uint64_t x, y, z; };
struct weft_l4 { uint64_t x, y, z, w; };
struct weft_ul2 { uint64_t x, y; };
struct weft_ul3 { uint64_t x, y, z; };
struct weft_ul4 { uint64_t x, y, z, w; };

// A load and a store through an address the kernel computed. Both front ends
// name these, so both targets have them.
template <class T>
static inline T weft_ld(const device void* p) { return *((const device T*)p); }
template <class T>
static inline void weft_st(device void* p, T v) { *((device T*)p) = v; }

#ifdef __METAL_VERSION__
// The GPU back end addresses threadgroup and per-thread memory the same way it
// addresses a buffer, so the load and store have to say which space they read.
// The emitter knows, and picks the overload; the CPU sees only the two above.
template <class T>
static inline T weft_ld(threadgroup const void* p) { return *((threadgroup const T*)p); }
template <class T>
static inline void weft_st(threadgroup void* p, T v) { *((threadgroup T*)p) = v; }
template <class T>
static inline T weft_ld(thread const void* p) { return *((thread const T*)p); }
template <class T>
static inline void weft_st(thread void* p, T v) { *((thread T*)p) = v; }
#endif

namespace weft {

// What a thread knows about itself: the same numbers the CPU runtime hands a
// lane, the buffers it was given, and the scalar parameters.
//
// A buffer is held as the *device pointer* the launcher bound, not as an
// address the host filled in: Metal's address space is the GPU's, so a host
// pointer means nothing inside a kernel (tools/ptrprobe answers that with the
// device's own words). The emitter's address arithmetic works on the uint64
// form of that pointer, which stays inside the GPU's address space when it is
// cast back, and that is the one representation both targets agree on.
#define WEFT_MAX_BUFS 32
struct Ctx {
  uint32_t lid;
  uint32_t tgid;
  uint32_t tg_size;
  uint32_t grid;   // total threads in the launch, as the CPU runtime reports it
  uint64_t gid;    // this thread's index in the launch, 64-bit like SPIR-V's
  device uint8_t* b[WEFT_MAX_BUFS];
  device uint64_t* params;
  // The words a 64-bit read-modify-write locks on, zeroed by the launcher before
  // every dispatch.
  device uint32_t* locks;
};

// How many lanes one SIMD group can hold here. The owner loop that serialises a
// 64-bit atomic across a warp runs this many steps, and the count has to come
// from the launch rather than be the literal 32: a fixed count with a spin loop
// inside it is something the Metal compiler tries to unroll, and on a loop of
// this shape it gives up and the pipeline is refused.
static inline uint32_t lane_span(thread const Ctx& c) {
  return c.tg_size < 32u ? c.tg_size : 32u;
}

static inline uint64_t buf_addr(thread const Ctx& c, int i) {
  return (uint64_t)(uintptr_t)c.b[i];
}

// How many threads of this group are real: the last group of a grid is usually
// short, and a ballot's member mask has to say so.
static inline uint32_t live_threads(uint32_t tgid, uint32_t tg_size, uint32_t grid) {
  const uint64_t base = (uint64_t)tgid * (uint64_t)tg_size;
  if (base >= (uint64_t)grid) return 0u;
  const uint64_t left = (uint64_t)grid - base;
  return (uint32_t)(left < (uint64_t)tg_size ? left : (uint64_t)tg_size);
}

// A declaration `OpVariable` with an initialiser zeroes threadgroup memory
// before the body runs; there is no memset here, so it is a store per byte.
static inline void zero_shared(threadgroup uint8_t* d, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) d[i] = 0u;
}
static inline void zero_thread(thread uint8_t* d, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) d[i] = 0u;
}

// ---- images ----
//
// An image is the bound texture and the two numbers a kernel asks about it. It
// travels by value because it has to: a texture is an opaque handle, and thread
// space cannot hold an address of one. The access mode is part of the type, so
// `ImageRef` is a sampled texture and `RWImageRef` a storage image; both take a
// `read`, which is what `OpImageRead` is.
struct ImageRef {
  texture2d<float, access::sample> t;
  uint32_t width;
  uint32_t height;
};
struct RWImageRef {
  texture2d<float, access::read_write> t;
  uint32_t width;
  uint32_t height;
};
struct SampledImage {
  ImageRef tex;
  sampler s;
};

static inline ImageRef make_image(texture2d<float, access::sample> t) {
  return ImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height()};
}
static inline RWImageRef make_rw_image(texture2d<float, access::read_write> t) {
  return RWImageRef{t, (uint32_t)t.get_width(), (uint32_t)t.get_height()};
}

// The launcher binds one-channel images, so a texel is channel 0 and the value
// is those four bytes. A read outside the image is nothing, which is what the
// hardware answers too.
// A reference has to name its address space in this language, and what a kernel
// holds is a copy of the handle in thread space.
template <class T, class R>
static inline T tex_read(thread const R& im, uint32_t x, uint32_t y) {
  if (x >= im.width || y >= im.height) return (T)0;
  return weft_bitcast<T>(im.t.read(uint2(x, y)).x);
}

// A sample goes through the bound sampler state, which is where the filter and
// the address mode live on this target: the CPU runtime models the same two
// choices, and the launcher hands them to both.
template <class T>
static inline T tex_sample(thread const SampledImage& si, float u, float v) {
  return weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v)).x);
}

// A texel is channel 0 of whatever the module wrote — the format is one channel,
// so the other three have nowhere to go, exactly as the CPU runtime copies only
// the first element.
static inline float tex_chan(float v) { return v; }
static inline float tex_chan(int32_t v) { return as_type<float>(v); }
static inline float tex_chan(uint32_t v) { return as_type<float>(v); }
static inline float tex_chan(int16_t v) { return (float)v; }
static inline float tex_chan(uint16_t v) { return (float)v; }

// The four channels a `tex` instruction returns. The binding is one channel
// wide, so the texel is channel 0 and the rest is the shape the format has.
template <class T>
static inline void tex_sample4(thread const SampledImage& si, float u, float v,
                               thread float* out) {
  out[0] = weft_bitcast<T>(si.tex.t.sample(si.s, float2(u, v)).x);
  out[1] = 0.0f;
  out[2] = 0.0f;
  out[3] = 1.0f;
}

// `tld4`: the four texels around the coordinate, each on its own channel,
// instead of the one a filter would blend. The footprint is the 2x2 block whose
// base is the texel under the coordinate, which is what the instruction means.
template <class T>
static inline void tex_gather4(thread const SampledImage& si, float u, float v,
                               thread float* out) {
  const int x0 = (int)floor(u * (float)si.tex.width - 0.5f);
  const int y0 = (int)floor(v * (float)si.tex.height - 0.5f);
  auto at = [&](int x, int y) -> float {
    if (x < 0 || y < 0 || (uint32_t)x >= si.tex.width || (uint32_t)y >= si.tex.height) return 0.0f;
    return weft_bitcast<T>(si.tex.t.read(uint2((uint32_t)x, (uint32_t)y)).x);
  };
  out[0] = at(x0, y0);
  out[1] = at(x0 + 1, y0);
  out[2] = at(x0, y0 + 1);
  out[3] = at(x0 + 1, y0 + 1);
}

template <class R, class V>
static inline void tex_write(thread const R& im, uint32_t x, uint32_t y, V v) {
  if (x >= im.width || y >= im.height) return;
  im.t.write(float4(tex_chan(v.x), 0.0f, 0.0f, 0.0f), uint2(x, y));
}

// SPIR-V says `OpAtomicLoad` / `OpAtomicStore` with a type; on the GPU those are
// the atomic instructions over the same 32 bits.
template <class T>
static inline T atomic_load(device void* p) {
  device atomic_uint* a = (device atomic_uint*)p;
  return as_type<T>(atomic_load_explicit(a, memory_order_relaxed));
}
template <class T>
static inline void atomic_store(device void* p, T v) {
  device atomic_uint* a = (device atomic_uint*)p;
  atomic_store_explicit(a, as_type<uint32_t>(v), memory_order_relaxed);
}
static inline uint64_t param(thread const Ctx& c, int i) { return c.params[i]; }

// ---- atomics ----
// The hardware's own fetch forms, not a retry loop, and that is not a
// micro-optimisation. A compare-exchange retry costs every contender one line
// transfer per success, so a counter touched by n threads costs about n^2 of them:
// at 4,096 lanes that is 16 million and at 65,536 it is 4 billion, which does not
// finish — it reads as a hang, and it did: the reduction path below (no lock, no
// 64-bit anything) hung at 65,536 lanes with the retry loop and finishes now.
// A CAS loop is only used where no instruction exists: min and max.
static inline uint32_t atom_add_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_add_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_sub_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_sub_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_and_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_and_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_or_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_or_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_xor_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_xor_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_exch_u32(device uint32_t* p, uint32_t v) {
  return atomic_exchange_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
// compare-exchange *is* the instruction for cas, and it hands back the old value
// whether or not it matched, which is the answer this has to return.
static inline uint32_t atom_cas_u32(device uint32_t* p, uint32_t cmp, uint32_t v) {
  uint32_t expect = cmp;
  (void)atomic_compare_exchange_weak_explicit((device atomic_uint*)p, &expect, v,
                                              memory_order_relaxed, memory_order_relaxed);
  return expect;
}
static inline int32_t atom_add_s32(device int32_t* p, int32_t v) {
  return (int32_t)atom_add_u32((device uint32_t*)p, (uint32_t)v);
}
// min and max have instructions here too, for both integer types — the note that
// they are refused at pipeline creation was about *float* atomics, and a probe
// asking for `atomic_fetch_max_explicit` on `atomic_uint` and `atomic_int`
// pipelines on this host. Using them matters for the same reason as add: a retry
// loop on one hot word costs n^2 line transfers, and 1,048,576 lanes is where that
// shows.
static inline int32_t atom_min_s32(device int32_t* p, int32_t v) {
  return atomic_fetch_min_explicit((device atomic_int*)p, v, memory_order_relaxed);
}
static inline int32_t atom_max_s32(device int32_t* p, int32_t v) {
  return atomic_fetch_max_explicit((device atomic_int*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_min_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_min_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
static inline uint32_t atom_max_u32(device uint32_t* p, uint32_t v) {
  return atomic_fetch_max_explicit((device atomic_uint*)p, v, memory_order_relaxed);
}
// A float add has an instruction as well (`atomic_float`), so it is native.
static inline float atom_add_f32(device float* p, float v) {
  return atomic_fetch_add_explicit((device atomic_float*)p, v, memory_order_relaxed);
}
// Float min and max do not: `atomic_fetch_min_explicit` / `max` are constrained to
// the integer types, and asking for `atomic_float` is a compile error. So this is
// the only retry loop left in the runtime, and it is the one place a very large grid
// could still grind — a hot word costs the number of contenders per success. Nothing
// in `kernels/` emits `atom.min.f32` or `atom.max.f32`; if something does, the pause
// below is what keeps it from spinning on the line. The arithmetic is done on the
// float the bits are, and `+.rn` is the addition this language performs.
static inline float cas_f32(device float* p, float v, int mode) {
  device atomic_uint* a = (device atomic_uint*)p;
  uint32_t old = atomic_load_explicit(a, memory_order_relaxed);
  uint32_t wait = 1u;
  for (;;) {
    const float o = as_type<float>(old);
    const float want = (mode == 1) ? min(o, v) : max(o, v);
    if (atomic_compare_exchange_weak_explicit(a, &old, as_type<uint32_t>(want),
                                              memory_order_relaxed, memory_order_relaxed))
      return o;
    for (uint32_t i = 0; i < wait; i++) (void)atomic_load_explicit(a, memory_order_relaxed);
    wait = wait < 512u ? wait * 2u : 512u;
  }
}
static inline float atom_min_f32(device float* p, float v) { return cas_f32(p, v, 1); }
static inline float atom_max_f32(device float* p, float v) { return cas_f32(p, v, 2); }

// ---- 64-bit reductions ----
//
// A `red` produces no result, so it does not need the 64 bits read as one value —
// it needs the pair to end up holding the fold of everything applied to it. For
// an operation that is order-free that is two 32-bit atomics:
//
//   add/sub  the low half goes first and its carry (or borrow) is applied to the
//            high half. Every thread that wraps the low word adds exactly one to
//            the high word, so the sum is right whatever order the threads land
//            in — and the hardware does the low half atomically, so nothing is
//            lost in between.
//   and/or/xor  each half is folded on its own. The two halves are independent,
//            so the pair ends up holding the fold without either half needing to
//            know about the other.
//
// What this cannot do is hand back the old pair as one value, or compare one.
// That needs the pair read as one value, which no instruction here does — so it
// is a lock over the two halves instead.

// ---- 64-bit read-modify-write ----
//
// There is no 64-bit atomic on this host. `atomic_ulong` is declared, but every
// entry point that could read, write, exchange or compare one is constrained to
// 32-bit types, and the two that are not (`atomic_max_explicit` /
// `atomic_min_explicit`) compile only to be refused by the device at pipeline
// creation — tools/atom64_probe.mm asks for each of those answers separately.
//
// So a 64-bit atomic is built out of the 32-bit ones the hardware does have: a
// lock on the word, held while the pair is read, changed and written back. The
// halves move through `atomic_load_explicit` / `atomic_store_explicit` rather
// than as one 64-bit load and store, because a plain load of a `device` word can
// be served from a core's own cache, which is not the point the atomics agree on.
//
// The wait spins, so no thread may be able to freeze the thread it waits for. Two
// threads in one SIMD group share a program counter, so a kernel that runs one
// hardware lane per thread takes turns first: only the lane whose id matches the
// current step is inside the critical section (the owner loop the translators
// emit), and then nobody waits on a peer that is itself waiting. A warp kernel
// has one thread per warp, so its lanes are data and never contend.
//
// A thread holds at most one lock at a time and a holder is by definition
// running, so it always reaches its release: nothing here can deadlock.
static constant uint32_t kLockWords = 1024u;

static inline device uint32_t* lock_of(device uint32_t* locks, device const void* p) {
  const uint64_t a = (uint64_t)(uintptr_t)p;
  const uint32_t h = (uint32_t)(a >> 3) ^ (uint32_t)(a >> 13);
  return locks + (h & (kLockWords - 1u));
}
// One word per lock, taken by test-and-set. A ticket lock would be fairer and was
// tried first: it deadlocks here, because a ticket is taken behind a thread whose
// threadgroup has not been dispatched yet, and that thread cannot reach its turn
// from off the device. So the wait is a plain spin, made polite: a loser stops
// writing and reads until the word looks free, and waits a jittered number of
// reads before trying again. Writing every time is what makes the line ping-pong
// between thousands of waiters, and retrying in step is what let one waiter lose
// for minutes.
static inline void lock_take(device uint32_t* l) {
  device atomic_uint* a = (device atomic_uint*)l;
  uint32_t wait = 1u;
  uint32_t expect = 0u;
  for (;;) {
    if (atomic_compare_exchange_weak_explicit(a, &expect, 1u, memory_order_relaxed,
                                             memory_order_relaxed))
      return;
    expect = 0u;
    uint32_t spins = wait + ((uint32_t)(uintptr_t)l & 7u);
    while (spins != 0u && atomic_load_explicit(a, memory_order_relaxed) != 0u) spins--;
    wait = wait < 64u ? wait * 2u : 64u;
  }
}
static inline void lock_drop(device uint32_t* l) {
  atomic_store_explicit((device atomic_uint*)l, 0u, memory_order_relaxed);
}

static inline uint64_t load64(device uint64_t* p) {
  device atomic_uint* a = (device atomic_uint*)p;
  const uint32_t lo = atomic_load_explicit(a, memory_order_relaxed);
  const uint32_t hi = atomic_load_explicit(a + 1, memory_order_relaxed);
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}
static inline int64_t load64(device int64_t* p) {
  return as_type<int64_t>(load64((device uint64_t*)p));
}
static inline void store64(device uint64_t* p, uint64_t v) {
  device atomic_uint* a = (device atomic_uint*)p;
  atomic_store_explicit(a, (uint32_t)v, memory_order_relaxed);
  atomic_store_explicit(a + 1, (uint32_t)(v >> 32), memory_order_relaxed);
}
static inline void store64(device int64_t* p, int64_t v) {
  store64((device uint64_t*)p, as_type<uint64_t>(v));
}

// A comparison and a select on the pair are written as two 32-bit words, and that
// is measured rather than stylistic. This host's pipeline compiler dies at
// creation (XPC_ERROR_CONNECTION_INTERRUPTED, 3/3, while the same helper doing an
// add pipelines) when a 64-bit compare or a 64-bit select sits inside a function
// that is inlined into a kernel; the identical code over 32-bit halves and a mask
// select pipelines. tools/msl_bisect.py walks the variants if the compiler ever
// changes.
static inline uint32_t gt_u64(uint64_t a, uint64_t b) {  // a > b, as 1 or 0
  const uint32_t a1 = (uint32_t)(a >> 32), b1 = (uint32_t)(b >> 32);
  const uint32_t a0 = (uint32_t)a, b0 = (uint32_t)b;
  return ((a1 > b1) | ((a1 == b1) & (a0 > b0))) ? 1u : 0u;
}
static inline uint32_t gt_s64(int64_t a, int64_t b) {  // signed: the high word too
  const int32_t a1 = (int32_t)((uint64_t)a >> 32), b1 = (int32_t)((uint64_t)b >> 32);
  const uint32_t a0 = (uint32_t)a, b0 = (uint32_t)b;
  return ((a1 > b1) | ((a1 == b1) & (a0 > b0))) ? 1u : 0u;
}
static inline uint32_t eq_bits64(uint64_t a, uint64_t b) {
  return (((uint32_t)a == (uint32_t)b) & ((uint32_t)(a >> 32) == (uint32_t)(b >> 32))) ? 1u : 0u;
}
// `v` when take is 1, else `o`.
static inline uint64_t pick_u64(uint64_t o, uint64_t v, uint32_t take) {
  const uint32_t m = 0u - take;
  const uint32_t lo = ((uint32_t)v & m) | ((uint32_t)o & ~m);
  const uint32_t hi = ((uint32_t)(v >> 32) & m) | ((uint32_t)(o >> 32) & ~m);
  return ((uint64_t)hi << 32) | (uint64_t)lo;
}
static inline int64_t pick_s64(int64_t o, int64_t v, uint32_t take) {
  return as_type<int64_t>(pick_u64(as_type<uint64_t>(o), as_type<uint64_t>(v), take));
}

// The update with the lock taken and released around it.
#define WEFT_RMW(name, T, expr)                                             \
  static inline T rmw_##name(device uint32_t* locks, device T* p, T v) {    \
    device uint32_t* l = lock_of(locks, p);                                 \
    lock_take(l);                                                           \
    const T o = load64(p);                                                  \
    store64(p, (expr));                                                     \
    lock_drop(l);                                                           \
    return o;                                                               \
  }
// ...and the same update for a caller that already holds one.
#define WEFT_RMW_RAW(name, T, expr)        \
  static inline T rmw_##name##_raw(device T* p, T v) { \
    const T o = load64(p);                \
    store64(p, (expr));                   \
    return o;                             \
  }
#define WEFT_RMW_ALL(MACRO)                        \
  MACRO(add, uint64_t, o + v)                      \
  MACRO(sub, uint64_t, o - v)                      \
  MACRO(min, uint64_t, pick_u64(o, v, gt_u64(o, v))) \
  MACRO(max, uint64_t, pick_u64(o, v, gt_u64(v, o))) \
  MACRO(and, uint64_t, o & v)                      \
  MACRO(or, uint64_t, o | v)                       \
  MACRO(xor, uint64_t, o ^ v)                      \
  MACRO(exch, uint64_t, v)                         \
  MACRO(min, int64_t, pick_s64(o, v, gt_s64(o, v))) \
  MACRO(max, int64_t, pick_s64(o, v, gt_s64(v, o))) \
  MACRO(add, int64_t, o + v)                       \
  MACRO(sub, int64_t, o - v)                       \
  MACRO(and, int64_t, o & v)                       \
  MACRO(or, int64_t, o | v)                        \
  MACRO(xor, int64_t, o ^ v)                       \
  MACRO(exch, int64_t, v)
WEFT_RMW_ALL(WEFT_RMW)
WEFT_RMW_ALL(WEFT_RMW_RAW)
#undef WEFT_RMW_ALL
#undef WEFT_RMW_RAW
#undef WEFT_RMW

// `atom.cas` keeps the word it found and only writes when the pair matches, so
// it needs the compare as well as the value.
static inline uint64_t rmw_cas(device uint32_t* locks, device uint64_t* p, uint64_t c,
                              uint64_t v) {
  device uint32_t* l = lock_of(locks, p);
  lock_take(l);
  const uint64_t o = load64(p);
  if (eq_bits64(o, c)) store64(p, v);
  lock_drop(l);
  return o;
}
static inline int64_t rmw_cas(device uint32_t* locks, device int64_t* p, int64_t c, int64_t v) {
  device uint32_t* l = lock_of(locks, p);
  lock_take(l);
  const int64_t o = load64(p);
  if (eq_bits64(as_type<uint64_t>(o), as_type<uint64_t>(c))) store64(p, v);
  lock_drop(l);
  return o;
}
static inline uint64_t rmw_cas_raw(device uint64_t* p, uint64_t c, uint64_t v) {
  const uint64_t o = load64(p);
  if (eq_bits64(o, c)) store64(p, v);
  return o;
}
static inline int64_t rmw_cas_raw(device int64_t* p, int64_t c, int64_t v) {
  const int64_t o = load64(p);
  if (eq_bits64(as_type<uint64_t>(o), as_type<uint64_t>(c))) store64(p, v);
  return o;
}
static inline device uint32_t* lo32(device uint64_t* p) { return (device uint32_t*)p; }
static inline device uint32_t* hi32(device uint64_t* p) { return (device uint32_t*)p + 1; }
static inline void red_add_u64(device uint64_t* p, uint64_t v) {
  const uint32_t vl = (uint32_t)v, vh = (uint32_t)(v >> 32);
  const uint32_t old = atom_add_u32(lo32(p), vl);
  atom_add_u32(hi32(p), vh + ((old + vl) < old ? 1u : 0u));
}
static inline void red_sub_u64(device uint64_t* p, uint64_t v) {
  const uint32_t vl = (uint32_t)v, vh = (uint32_t)(v >> 32);
  const uint32_t old = atom_sub_u32(lo32(p), vl);
  atom_sub_u32(hi32(p), vh + (old < vl ? 1u : 0u));
}
static inline void red_and_u64(device uint64_t* p, uint64_t v) {
  atom_and_u32(lo32(p), (uint32_t)v);
  atom_and_u32(hi32(p), (uint32_t)(v >> 32));
}
static inline void red_or_u64(device uint64_t* p, uint64_t v) {
  atom_or_u32(lo32(p), (uint32_t)v);
  atom_or_u32(hi32(p), (uint32_t)(v >> 32));
}
static inline void red_xor_u64(device uint64_t* p, uint64_t v) {
  atom_xor_u32(lo32(p), (uint32_t)v);
  atom_xor_u32(hi32(p), (uint32_t)(v >> 32));
}

// ---- a reduction barrier ----
// `bar.red` publishes one bit per thread, then every thread folds the group.
// The barrier is the hardware's, so this is a publish, a barrier, and a loop —
// no rendezvous bookkeeping, which is what the CPU side has to do by hand.
static inline uint32_t red_fold(threadgroup uint32_t* slots, thread const Ctx& c, int mode) {
  uint32_t t = 0u;
  if (mode == 0) {
    for (uint32_t i = 0; i < c.tg_size; i++) t += (uint32_t)popcount(slots[i]);
  } else if (mode == 1) {
    t = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < c.tg_size; i++) t &= slots[i];
  } else {
    for (uint32_t i = 0; i < c.tg_size; i++) t |= slots[i];
  }
  return t;
}

// `cp.async` copies into threadgroup memory. The GPU has a real asynchronous
// copy, but the answer after the wait is the same bytes, so the copy is issued
// inline and the commit/wait stay no-ops.
static inline void copy_shared(threadgroup uint8_t* dst, const device void* src, uint32_t n) {
  const device uint8_t* s = (const device uint8_t*)src;
  for (uint32_t i = 0; i < n; i++) dst[i] = s[i];
}

}  // namespace weft
)WEFMSL";
