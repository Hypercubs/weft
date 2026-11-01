"""A reference for the wgmma sample: the tile product, computed from the inputs.

The kernel stores A and B into shared memory in the core-matrix layout its
descriptor describes, and the instruction reads them back through that
descriptor. Computing A[r][k] * B[k][n] straight from the input tiles is
therefore independent of both: a mismatch means the descriptor decode, the
core-matrix walk, or the accumulator pattern is wrong. The inputs are small
integers, exactly representable in f16, so every sum of 16 products is exact in
f32 as well.
"""
import sys, struct

def gen(pa, pb):
    # A[r][k] = -2..2 and B[k][n] = -3..3, both exactly representable in f16.
    A = [((r * 3 + k * 7) % 5) - 2 for r in range(64) for k in range(16)]
    B = [((k * 5 + n * 3) % 7) - 3 for k in range(16) for n in range(8)]
    open(pa, 'wb').write(struct.pack('<%de' % len(A), *A))
    open(pb, 'wb').write(struct.pack('<%de' % len(B), *B))

def check(pa, pb, pd):
    A = struct.unpack('<1024e', open(pa, 'rb').read(2048))
    B = struct.unpack('<128e', open(pb, 'rb').read(256))
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
    if sys.argv[1] == 'gen':
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(0 if check(*sys.argv[2:5]) else 1)
