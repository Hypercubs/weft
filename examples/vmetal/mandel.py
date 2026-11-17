"""The virtual Metal device, driven from Python, with no GPU.

This talks to libvmetal through ctypes, which is the C ABI in runtime/vmetal.h
reached the way any other language would reach it: no Metal, no Objective-C, no
weft binary, no GPU. The module it opens is a shared library that weft-gen
translated a .metal file into and this host's compiler built, so the arithmetic
that runs is the kernel's own body.

    python3 examples/vmetal/mandel.py build/libvmetal.dylib build/vmetal_mandel.dylib

The picture is 8192 words of escape-time Mandelbrot, and it is printed here as
text so the output is the picture itself rather than a number about it. The same
kernel on the M1's GPU is compared against its CPU column by `make
vmetal-picture`; this run is the third column, and it is the one with no device.
"""
import ctypes
import struct
import sys

W, H = 128, 64
GRID = W * H
THREADS = 64
RAMP = " .:-=+*#%@"

def bind(lib):
    lib.vmetal_device_name.restype = ctypes.c_char_p
    lib.vmetal_last_error.restype = ctypes.c_char_p
    lib.vmetal_device_open.restype = ctypes.c_void_p
    lib.vmetal_device_open.argtypes = [ctypes.c_char_p]
    lib.vmetal_device_release.argtypes = [ctypes.c_void_p]
    lib.vmetal_kernel_count.restype = ctypes.c_int
    lib.vmetal_kernel_count.argtypes = [ctypes.c_void_p]
    lib.vmetal_kernel_find.restype = ctypes.c_int
    lib.vmetal_kernel_find.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vmetal_kernel_buffers.restype = ctypes.c_int
    lib.vmetal_kernel_buffers.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vmetal_new_buffer.restype = ctypes.c_void_p
    lib.vmetal_new_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.vmetal_buffer_contents.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.vmetal_buffer_contents.argtypes = [ctypes.c_void_p]
    lib.vmetal_new_queue.restype = ctypes.c_void_p
    lib.vmetal_new_queue.argtypes = [ctypes.c_void_p]
    lib.vmetal_encoder_create.restype = ctypes.c_void_p
    lib.vmetal_encoder_create.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vmetal_encoder_set_buffer.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.vmetal_encoder_dispatch.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.vmetal_encoder_commit.restype = ctypes.c_int
    lib.vmetal_encoder_commit.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.vmetal_encoder_release.argtypes = [ctypes.c_void_p]
    return lib

def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: %s <libvmetal> <module>\n" % argv[0])
        return 2
    lib = bind(ctypes.CDLL(argv[1]))

    dev = lib.vmetal_device_open(argv[2].encode())
    if not dev:
        sys.stderr.write("vmetal: %s\n" % lib.vmetal_last_error().decode())
        return 1
    k = lib.vmetal_kernel_find(dev, b"mandel")
    if k < 0:
        sys.stderr.write("vmetal: %s\n" % lib.vmetal_last_error().decode())
        return 1

    # The module says how many buffers the kernel declares; binding more or
    # fewer is an error here, so the count comes from the module, not from this
    # file's idea of the signature.
    nbuf = lib.vmetal_kernel_buffers(dev, k)
    seed = (ctypes.c_float * GRID)()
    for i in range(GRID):
        seed[i] = 0.5 * float((i + 1) % 7) + 0.25
    out_init = (ctypes.c_float * GRID)()
    bufs = [lib.vmetal_new_buffer(dev, GRID * 4, ctypes.cast(seed, ctypes.c_void_p)),
            lib.vmetal_new_buffer(dev, GRID * 4, ctypes.cast(out_init, ctypes.c_void_p))]
    if nbuf > len(bufs):
        sys.stderr.write("vmetal: kernel declares %d buffers, this script has %d\n"
                         % (nbuf, len(bufs)))
        return 1

    q = lib.vmetal_new_queue(dev)
    enc = lib.vmetal_encoder_create(q, k)
    for i in range(nbuf):
        lib.vmetal_encoder_set_buffer(enc, i, bufs[i], 0)
    lib.vmetal_encoder_dispatch(enc, GRID, THREADS)
    ms = ctypes.c_double(0.0)
    if lib.vmetal_encoder_commit(enc, ctypes.byref(ms)) != 0:
        sys.stderr.write("vmetal: %s\n" % lib.vmetal_last_error().decode())
        return 1
    lib.vmetal_encoder_release(enc)

    raw = lib.vmetal_buffer_contents(bufs[1])
    words = ctypes.cast(raw, ctypes.POINTER(ctypes.c_uint32))
    counts = [struct.unpack("<f", struct.pack("<I", words[i]))[0] for i in range(GRID)]
    digest = 1469598103934665603
    for i in range(GRID):
        digest = ((digest ^ words[i]) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    print("device  %s" % lib.vmetal_device_name().decode())
    print("kernel  mandel  %dx%d pixels  %d threads  %.3f ms  python %d.%d"
          % (W, H, GRID, ms.value, sys.version_info[0], sys.version_info[1]))

    lines = []
    for y in range(H):
        row = []
        for x in range(W):
            # Dense characters are the pixels that never escaped, so the blob
            # is the set itself and the space around it is the escape region.
            n = counts[y * W + x]
            row.append(RAMP[min(len(RAMP) - 1, int(n * len(RAMP) / 128.0))])
        lines.append("".join(row))
    print("\n".join(lines))

    print("words   %d  fnv %016x  min %.0f  max %.0f  no gpu opened"
          % (GRID, digest, min(counts), max(counts)))
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
