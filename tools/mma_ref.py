#!/usr/bin/env python3
"""The reference for the shapes in `kernels/mma8.ptx`.

Each of those is one warp-wide dot product, and the fragment layouts are the ones
the standard states, so what this compares is the map from (lane, register,
element) to (row, column). Every input is a small integer, so each product and each
sum is exact in binary32 and in binary64, a difference is a fragment-map bug and
never a rounding one.

  k8f16   mma.m16n8k8: A is 16x8 halves, row major; Bt is 8x8 halves and holds B
          transposed (Bt[n][k] = B[k][n]); D is 16x8 floats.
  k4tf32  mma.m16n8k4: A is 16x4 floats, B is 4x8 floats, D is 16x8 floats. Both
          operands are integers, so the tf32 reduction changes nothing.
  s8      mma.m8n8k16: A is 8x16 bytes, Bt is 8x16 bytes, D is 8x8 int32.
  k64s4   mma.m16n8k64: A is 16x64 signed nibbles, Bt is 8x64, D is 16x8 int32.
          Two nibbles a byte, low first, each sign-extended from four bits.
"""
import struct
import sys


def read(path, fmt, count):
    data = open(path, 'rb').read()
    size = struct.calcsize(fmt)
    if len(data) < count * size:
        raise SystemExit('%s holds %d bytes and %d are needed' % (path, len(data), count * size))
    return list(struct.unpack('<%d%s' % (count, fmt), data[:count * size]))


def read_nibbles(path, count):
    """Signed four-bit elements, two to a byte, the low one first."""
    if count % 2:
        raise SystemExit('an odd number of nibbles has no byte packing')
    data = open(path, 'rb').read()
    if len(data) < count // 2:
        raise SystemExit('%s holds %d bytes and %d are needed' % (path, len(data), count // 2))
    out = []
    for i in range(count):
        byte = data[i // 2]
        n = (byte & 0xF) if i % 2 == 0 else (byte >> 4)
        out.append(n - 16 if n > 7 else n)
    return out


def read_nibbles_u(path, count):
    """Four-bit codes, two to a byte, the low one first, with no sign applied: the
    code of an fp4 is a sign bit and three magnitude bits, not a two's complement."""
    data = open(path, 'rb').read()
    if len(data) < count // 2:
        raise SystemExit('%s holds %d bytes and %d are needed' % (path, len(data), count // 2))
    out = []
    for i in range(count):
        byte = data[i // 2]
        out.append((byte & 0xF) if i % 2 == 0 else (byte >> 4))
    return out


def mx(code, eb, mb):
    """One narrow float, decoded the way `weft_mx_decode` decodes it: a sign bit above
    the exponent, subnormals on the same grid when the exponent is zero."""
    bias = (1 << (eb - 1)) - 1
    mant = code & ((1 << mb) - 1)
    e = (code >> mb) & ((1 << eb) - 1)
    ex = (1 if e == 0 else e) - bias - mb
    d = ((0 if e == 0 else (1 << mb)) + mant) * (2.0 ** ex)
    return -d if (code >> (eb + mb)) else d


def matmul(a, bt, rows, cols, k, transposed):
    out = []
    for r in range(rows):
        for n in range(cols):
            s = 0
            for i in range(k):
                b = bt[n * k + i] if transposed else bt[i * cols + n]
                s += a[r * k + i] * b
            out.append(s)
    return out


def main():
    if len(sys.argv) != 5:
        raise SystemExit('usage: mma_ref.py <k8f16|k4tf32|s8|k64s4|k32e2m1|k32e3m2|k64e2m1> <a> <bt> <d>')
    shape, apath, bpath, dpath = sys.argv[1:5]
    if shape == 'k8f16':
        want = matmul(read(apath, 'e', 16 * 8), read(bpath, 'e', 8 * 8), 16, 8, 8, True)
        got = read(dpath, 'f', 16 * 8)
    elif shape == 'k4tf32':
        want = matmul(read(apath, 'f', 16 * 4), read(bpath, 'f', 4 * 8), 16, 8, 4, False)
        got = read(dpath, 'f', 16 * 8)
    elif shape == 's8':
        want = matmul(read(apath, 'b', 8 * 16), read(bpath, 'b', 8 * 16), 8, 8, 16, True)
        got = read(dpath, 'i', 8 * 8)
    elif shape == 'k64s4':
        want = matmul(read_nibbles(apath, 16 * 64), read_nibbles(bpath, 8 * 64),
                      16, 8, 64, True)
        got = read(dpath, 'i', 16 * 8)
    elif shape == 'k32e2m1':
        want = matmul([mx(c, 2, 1) for c in read_nibbles_u(apath, 16 * 32)],
                      [mx(c, 2, 1) for c in read_nibbles_u(bpath, 8 * 32)], 16, 8, 32, True)
        got = read(dpath, 'f', 16 * 8)
    elif shape == 'k32e3m2':
        want = matmul([mx(c, 3, 2) for c in read(apath, 'B', 16 * 32)],
                      [mx(c, 3, 2) for c in read(bpath, 'B', 8 * 32)], 16, 8, 32, True)
        got = read(dpath, 'f', 16 * 8)
    elif shape == 'k64e2m1':
        want = matmul([mx(c, 2, 1) for c in read_nibbles_u(apath, 16 * 64)],
                      [mx(c, 2, 1) for c in read_nibbles_u(bpath, 8 * 64)], 16, 8, 64, True)
        got = read(dpath, 'f', 16 * 8)
    else:
        raise SystemExit('unknown shape ' + shape)

    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
    if bad:
        for i, w, g in bad[:8]:
            print('  element %d (row %d, column %d): wanted %g, read %g'
                  % (i, i // 8, i % 8, w, g))
        print('%s: %d of %d elements differ' % (shape, len(bad), len(want)))
        return 1
    print('%s: %d values match' % (shape, len(want)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
