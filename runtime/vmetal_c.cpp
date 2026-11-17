// The C ABI over runtime/vmetal.hpp, one wrapper per object.
#include "vmetal.h"

#include "vmetal.hpp"

#include <string>
#include <utility>

struct vmetal_device {
  vmetal::Device d;
};

struct vmetal_buffer {
  vmetal::Buffer b;
};

struct vmetal_texture {
  vmetal::Texture t;
};

struct vmetal_queue {
  vmetal::CommandQueue q;
};

struct vmetal_encoder {
  vmetal::CommandBuffer cb;
  vmetal::Encoder enc;
  explicit vmetal_encoder(vmetal::CommandBuffer c) : cb(std::move(c)), enc(cb.compute_encoder()) {}
};

namespace {
thread_local std::string g_error;

void fail(const std::string& why) { g_error = why; }
void clear() { g_error.clear(); }
}  // namespace

extern "C" {

const char* vmetal_device_name(void) { return vmetal::device_name(); }

const char* vmetal_last_error(void) { return g_error.c_str(); }

vmetal_device* vmetal_device_open(const char* module_path) {
  std::string err;
  vmetal::Device d = vmetal::Device::open(module_path ? module_path : "", &err);
  if (!d.valid()) {
    fail(err);
    return nullptr;
  }
  clear();
  return new vmetal_device{std::move(d)};
}

static vmetal::SourceKind kind_from_int(int kind) {
  switch (kind) {
    case 1: return vmetal::SourceKind::Metal;
    case 2: return vmetal::SourceKind::PTX;
    case 3: return vmetal::SourceKind::SPIRV;
    default: return vmetal::SourceKind::Auto;
  }
}

vmetal_device* vmetal_device_from_source(const char* source) {
  return vmetal_device_from_source_kind(source, 0);
}

static vmetal_device* device_from(vmetal::Library& lib, std::string& err) {
  if (!lib.valid()) {
    fail(err);
    return nullptr;
  }
  vmetal::Device d = lib.device();
  if (!d.valid()) {
    fail("the compiled module made no device");
    return nullptr;
  }
  clear();
  return new vmetal_device{std::move(d)};
}

vmetal_device* vmetal_device_from_source_kind(const char* source, int kind) {
  std::string err;
  vmetal::Library lib =
      vmetal::Device::new_library(source ? source : "", kind_from_int(kind), &err);
  return device_from(lib, err);
}

vmetal_device* vmetal_device_from_source_buffers(const char* source, int kind, const int* slots,
                                                 const unsigned int* widths, int count) {
  vmetal::LibraryOptions opts;
  opts.kind = kind_from_int(kind);
  for (int i = 0; slots && widths && i < count; i++)
    if (slots[i] >= 0 && widths[i]) opts.widths.push_back({slots[i], widths[i]});
  std::string err;
  vmetal::Library lib = vmetal::Device::new_library(source ? source : "", opts, &err);
  return device_from(lib, err);
}

void vmetal_device_release(vmetal_device* d) { delete d; }

int vmetal_kernel_count(vmetal_device* d) { return d ? d->d.kernel_count() : 0; }

int vmetal_kernel_find(vmetal_device* d, const char* name) {
  if (!d || !name) {
    fail("no device, or no kernel name");
    return -1;
  }
  std::string err;
  const vmetal::PipelineState p = d->d.new_pipeline(name, &err);
  if (!p.valid()) {
    fail(err);
    return -1;
  }
  clear();
  return p.index;
}

int vmetal_kernel_buffers(vmetal_device* d, int kernel_index) {
  if (!d) return 0;
  const weft_kernel_info* k = d->d.kernel_info(kernel_index);
  return k ? k->nbufs : 0;
}

vmetal_buffer* vmetal_new_buffer(vmetal_device* d, size_t bytes, const void* init) {
  if (!d) {
    fail("no device");
    return nullptr;
  }
  return new vmetal_buffer{d->d.new_buffer(bytes, init)};
}

unsigned char* vmetal_buffer_contents(vmetal_buffer* b) { return b ? b->b.contents() : nullptr; }

size_t vmetal_buffer_length(vmetal_buffer* b) { return b ? b->b.length() : 0; }

void vmetal_buffer_release(vmetal_buffer* b) { delete b; }

vmetal_texture* vmetal_new_texture(vmetal_device* d, uint32_t width, uint32_t height,
                                   uint32_t layers, uint32_t elem, const void* init) {
  if (!d) {
    fail("no device");
    return nullptr;
  }
  return new vmetal_texture{d->d.new_texture(width, height, layers, elem, init)};
}

unsigned char* vmetal_texture_contents(vmetal_texture* t) { return t ? t->t.contents() : nullptr; }

void vmetal_texture_release(vmetal_texture* t) { delete t; }

vmetal_queue* vmetal_new_queue(vmetal_device* d) {
  if (!d) {
    fail("no device");
    return nullptr;
  }
  return new vmetal_queue{d->d.new_queue()};
}

void vmetal_queue_release(vmetal_queue* q) { delete q; }

vmetal_encoder* vmetal_encoder_create(vmetal_queue* q, int kernel_index) {
  if (!q) {
    fail("no queue");
    return nullptr;
  }
  vmetal::CommandBuffer cb = q->q.command_buffer();
  cb.set_pipeline(vmetal::PipelineState{kernel_index, std::string()});
  return new vmetal_encoder(std::move(cb));
}

void vmetal_encoder_set_buffer(vmetal_encoder* e, int index, vmetal_buffer* b, size_t offset) {
  if (!e || !b) return;
  e->enc.set_buffer(index, b->b, offset);
}

void vmetal_encoder_set_texture(vmetal_encoder* e, int index, vmetal_texture* t) {
  if (!e || !t) return;
  e->enc.set_texture(index, t->t);
}

void vmetal_encoder_set_sampler(vmetal_encoder* e, int index, int filter, int address) {
  if (!e) return;
  e->enc.set_sampler(index, filter, address);
}

void vmetal_encoder_dispatch(vmetal_encoder* e, uint32_t grid, uint32_t threads) {
  if (!e) return;
  e->enc.dispatch_threads(grid, threads);
}

void vmetal_encoder_dispatch_indirect(vmetal_encoder* e, vmetal_buffer* args, size_t offset,
                                      uint32_t threads) {
  if (!e || !args) return;
  e->enc.dispatch_threadgroups_indirect(args->b, offset, threads);
}

void vmetal_encoder_fill_buffer(vmetal_encoder* e, vmetal_buffer* b, size_t offset, size_t bytes,
                                unsigned char value) {
  if (!e || !b) return;
  e->enc.fill_buffer(b->b, offset, bytes, value);
}

void vmetal_encoder_copy_buffer(vmetal_encoder* e, vmetal_buffer* src, size_t src_offset,
                                vmetal_buffer* dst, size_t dst_offset, size_t bytes) {
  if (!e || !src || !dst) return;
  e->enc.copy_buffer(src->b, src_offset, dst->b, dst_offset, bytes);
}

int vmetal_encoder_commit(vmetal_encoder* e, double* elapsed_ms) {
  if (!e) {
    fail("no encoder");
    return 1;
  }
  e->enc.end_encoding();
  e->cb.commit();
  e->cb.wait_until_completed();
  if (!e->cb.ok()) {
    fail(e->cb.error());
    return 1;
  }
  clear();
  if (elapsed_ms) *elapsed_ms = e->cb.elapsed_ms();
  return 0;
}

void vmetal_encoder_release(vmetal_encoder* e) { delete e; }

}  // extern "C"
