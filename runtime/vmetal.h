/* The virtual Metal device as a C ABI.
 *
 * This is the same device runtime/vmetal.hpp exposes to C++: open a module,
 * find a kernel, allocate buffers and textures, record a dispatch on an encoder
 * and read the results back. Nothing here is Metal's own ABI; it is a small
 * stable surface a language binding can call through a shared library.
 *
 *     vmetal_device* d = vmetal_device_open("kernel.dylib");
 *     int k = vmetal_kernel_find(d, "saxpy");
 *     vmetal_buffer* a = vmetal_new_buffer(d, bytes, init);
 *     vmetal_encoder* e = vmetal_encoder_create(vmetal_new_queue(d), k);
 *     vmetal_encoder_set_buffer(e, 0, a, 0);
 *     vmetal_encoder_dispatch(e, grid, threads);
 *     vmetal_encoder_commit(e, &ms);
 *
 * Every call that can fail returns NULL or a non-zero int, and the reason is in
 * vmetal_last_error, which is thread-local.
 */
#ifndef VMETAL_H
#define VMETAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vmetal_device vmetal_device;
typedef struct vmetal_buffer vmetal_buffer;
typedef struct vmetal_texture vmetal_texture;
typedef struct vmetal_queue vmetal_queue;
typedef struct vmetal_encoder vmetal_encoder;

/* The name every device reports: "weft virtual Metal". */
const char* vmetal_device_name(void);

/* The reason the last call on this thread failed, or "" when it did not. */
const char* vmetal_last_error(void);

/* Load a module built by weft-gen plus this host's compiler. */
vmetal_device* vmetal_device_open(const char* module_path);
/* Compile shader source at run time, the way newLibraryWithSource: does: it is
 * translated to the same C++ the build-time path writes and this host's compiler
 * builds it. WEFT_GEN names weft-gen, WEFT_CXX the compiler and WEFT_ROOT the tree
 * with include/ and runtime/. NULL when no toolchain answers, with the reason in
 * vmetal_last_error. */
vmetal_device* vmetal_device_from_source(const char* source);
/* The same, with the language named: 0 reads it, 1 Metal, 2 PTX, 3 SPIR-V. */
vmetal_device* vmetal_device_from_source_kind(const char* source, int kind);
/* The same, with element widths for the slots the source does not name: slot
 * slots[i] has widths[i] bytes. count is how many pairs there are. */
vmetal_device* vmetal_device_from_source_buffers(const char* source, int kind, const int* slots,
                                                 const unsigned int* widths, int count);
void vmetal_device_release(vmetal_device*);

int vmetal_kernel_count(vmetal_device*);
/* The kernel's index in the module, or -1 when it has no such kernel. */
int vmetal_kernel_find(vmetal_device*, const char* name);
/* The buffers the kernel declares, the count an encoder has to bind. */
int vmetal_kernel_buffers(vmetal_device*, int kernel_index);

/* A buffer is `bytes` long and starts as `init` when that is not NULL. */
vmetal_buffer* vmetal_new_buffer(vmetal_device*, size_t bytes, const void* init);
unsigned char* vmetal_buffer_contents(vmetal_buffer*);
size_t vmetal_buffer_length(vmetal_buffer*);
void vmetal_buffer_release(vmetal_buffer*);

/* A texture is `layers` images of width by height, `elem` bytes a texel. */
vmetal_texture* vmetal_new_texture(vmetal_device*, uint32_t width, uint32_t height,
                                   uint32_t layers, uint32_t elem, const void* init);
unsigned char* vmetal_texture_contents(vmetal_texture*);
void vmetal_texture_release(vmetal_texture*);

vmetal_queue* vmetal_new_queue(vmetal_device*);
void vmetal_queue_release(vmetal_queue*);

/* An encoder records dispatches and the bindings they run with. A binding holds
 * for every dispatch encoded after it, and each dispatch call adds one dispatch
 * to the command buffer, run in the order they were encoded. */
vmetal_encoder* vmetal_encoder_create(vmetal_queue*, int kernel_index);
void vmetal_encoder_set_buffer(vmetal_encoder*, int index, vmetal_buffer*, size_t offset);
void vmetal_encoder_set_texture(vmetal_encoder*, int index, vmetal_texture*);
/* filter: 0 nearest, 1 linear.  address: 0 clamp to edge, 1 clamp to zero, 2 repeat. */
void vmetal_encoder_set_sampler(vmetal_encoder*, int index, int filter, int address);
void vmetal_encoder_dispatch(vmetal_encoder*, uint32_t grid, uint32_t threads);
/* The group counts are three 32-bit words at `offset` in `args`: x, y, z. This
 * engine is a line of threads, so y or z other than one is an error at commit. */
void vmetal_encoder_dispatch_indirect(vmetal_encoder*, vmetal_buffer* args, size_t offset,
                                      uint32_t threads);
/* Blits take their place in the same ordered command list as a dispatch. */
void vmetal_encoder_fill_buffer(vmetal_encoder*, vmetal_buffer*, size_t offset, size_t bytes,
                                unsigned char value);
void vmetal_encoder_copy_buffer(vmetal_encoder*, vmetal_buffer* src, size_t src_offset,
                                vmetal_buffer* dst, size_t dst_offset, size_t bytes);
/* Runs every dispatch encoded on the buffer, in order. Returns 0 on success, 1 on
 * failure, and writes the wall time in milliseconds to `elapsed_ms` when that is
 * not NULL. A slot the kernel reads but nothing bound, or one bound past its own
 * length, fails here before anything runs. The encoder is spent either way and is
 * released with vmetal_encoder_release. */
int vmetal_encoder_commit(vmetal_encoder*, double* elapsed_ms);
void vmetal_encoder_release(vmetal_encoder*);

#ifdef __cplusplus
}
#endif

#endif /* VMETAL_H */
