"""A reference for the wgmma samples: the tile product, computed from the inputs.

The kernel stores A and B into shared memory in the core-matrix layout its
descriptor describes, and the instruction reads them back through that
descriptor. Computing A[r][k] * B[k][n] straight from the input tiles is
therefore independent of both: a mismatch means the descriptor decode, the
core-matrix walk, or the accumulator pattern is wrong. The inputs are small
integers, exactly representable in both f16 and bf16, so every sum of 16 products
is exact in f32 as well.

The trailing `bf16` argument reads and writes bfloat16 instead of f16. The two
types are the same width and the same layout, which is why one translator reads
either — and different numbers for the same bits, which is why this reference can
tell one decode from the other.
"""
import sys, struct


def _pack(v, bf16):
    if not bf16:
        return struct.pack('<e', v)
    return struct.pack('<H', struct.unpack('<I', struct.pack('<f', float(v)))[0] >> 16)


def _unpack(b, bf16):
    if not bf16:
        return struct.unpack('<e', b)[0]
    return struct.unpack('<f', struct.pack('<I', struct.unpack('<H', b)[0] << 16))[0]


def gen(pa, pb, bf16=False):
    # A[r][k] = -2..2 and B[k][n] = -3..3, both exactly representable in f16 and
    # in bf16.
    A = [((r * 3 + k * 7) % 5) - 2 for r in range(64) for k in range(16)]
    B = [((k * 5 + n * 3) % 7) - 3 for k in range(16) for n in range(8)]
    open(pa, 'wb').write(b''.join(_pack(v, bf16) for v in A))
    open(pb, 'wb').write(b''.join(_pack(v, bf16) for v in B))


def check(pa, pb, pd, bf16=False):
    araw = open(pa, 'rb').read(2048)
    braw = open(pb, 'rb').read(256)
    A = [_unpack(araw[2 * i:2 * i + 2], bf16) for i in range(1024)]
    B = [_unpack(braw[2 * i:2 * i + 2], bf16) for i in range(128)]
    D = struct.unpack('<512f', open(pd, 'rb').read(2048))
    bad = 0
    for r in range(64):
        for n in range(8):
            want = float(sum(A[r * 16 + k] * B[k * 8 + n] for k in range(16)))
            got = D[r * 8 + n]
            if got != want:
                if bad < 4:
                    print("  D[%d][%d] want %r got %r" % (r, n, want, got))
                bad += 1
    print("  %d of 512 accumulator elements differ" % bad)
    return bad == 0


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    bf16 = bool(args) and args[-1] == 'bf16'
    if bf16:
        args = args[:-1]
    if mode == 'gen':
        gen(args[0], args[1], bf16)
    else:
        sys.exit(0 if check(args[0], args[1], args[2], bf16) else 1)
