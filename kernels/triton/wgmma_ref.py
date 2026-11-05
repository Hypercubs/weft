"""A reference for the wgmma samples: the tile product, computed from the inputs.

The kernel stores A and B into shared memory in the core-matrix layout its
descriptor describes, and the instruction reads them back through that
descriptor. Computing A[r][k] * B[k][n] straight from the input tiles is
therefore independent of both: a mismatch means the descriptor decode, the
core-matrix walk, or the accumulator pattern is wrong.

The last argument picks the tile: the element type, which selects K because a
core-matrix row is 16 bytes and holds 16/element_size values, and how many
columns of the result the tile covers.

  (nothing)  f16,  K = 16, two bytes an element, eight columns
  f16n16            the same element over sixteen columns, two N groups, so B's
                    SBO term is a real address and a lane holds eight
                    accumulators instead of four
  f16acc            the same tile accumulated twice, so the expected value is two
                    products: the only mode where scale-d can be observed
  f16chain          issued three times, which is where an accumulator that is read
                    once and then dropped still has to carry: three products
  bf16              the same width and layout as f16, different numbers for the
                    same bits, how one translator reading either is told from
                    one that assumes f16
  tf32       K = 8, four bytes: the instruction truncates to 19 bits, and the
                    inputs are integers, so truncation changes nothing
  e4m3       K = 32, one byte: mantissa 3, exponent 4, saturating at 448
  e5m2              the same width, mantissa 2, exponent 5, with infinities
  s8         K = 32, one byte, signed, the same bytes as a u8 tile are
  u8                different numbers, so one reference tells the two apart even
                    though the layout is byte-for-byte identical
  s8acc      the same bytes with an s32 accumulator, the other pair the
  u8acc     instruction offers: the answer is an integer, and it is read as one

Every input is a value that is exact in the format, so each sum of K products is
exact: a difference is a layout error, not a rounding one.
"""
import os, sys, struct
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_ref

# k: the instruction's K. w: bytes per element. fmt: how a value is encoded.
# cols: how many columns of the result the tile covers.
SPECS = {
    'f16':    dict(k=16, w=2, fmt='f16',  cols=8),
    'f16n16': dict(k=16, w=2, fmt='f16',  cols=16),
    'f16acc': dict(k=16, w=2, fmt='f16',  cols=8, mult=2),
    'f16chain': dict(k=16, w=2, fmt='f16', cols=8, mult=3),
    'f16neg': dict(k=16, w=2, fmt='f16', cols=8, mult=-1),
    'bf16':   dict(k=16, w=2, fmt='bf16', cols=8),
    'tf32':   dict(k=8,  w=4, fmt='tf32', cols=8),
    'e4m3':   dict(k=32, w=1, fmt='e4m3', cols=8),
    'e5m2':   dict(k=32, w=1, fmt='e5m2', cols=8),
    's8':     dict(k=32, w=1, fmt='s8',   cols=8),
    'u8':     dict(k=32, w=1, fmt='u8',   cols=8),
    's8acc':  dict(k=32, w=1, fmt='s8',   cols=8, iacc=True),
    'u8acc':  dict(k=32, w=1, fmt='u8',   cols=8, iacc=True),
}

ROWS = 64

# Values exact in every float format above and ordinary integers in the integer
# ones. The integer pools stay inside what both signedness can hold, so the same
# bytes remain meaningful under either reading.
POOL = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, -0.5, -1.0, -2.0, 0.25, 0.0, 2.5]
POOL_I8 = [-4, -2, -1, 0, 1, 2, 3, 5]
POOL_U8 = [0, 1, 2, 3, 4, 5, 7, 9]


def _pool(fmt):
    if fmt == 's8':
        return POOL_I8
    if fmt == 'u8':
        return POOL_U8
    return POOL


def _enc(v, fmt):
    if fmt in ('e4m3', 'e5m2'):
        return bytes([fp8_ref.enc(v, fp8_ref.FMT[fmt])])
    if fmt in ('s8', 'u8'):
        return bytes([int(v) & 0xFF])
    if fmt == 'bf16':
        return struct.pack('<H', struct.unpack('<I', struct.pack('<f', float(v)))[0] >> 16)
    return struct.pack('<e' if fmt == 'f16' else '<f', v)


def _dec(b, fmt):
    if fmt in ('e4m3', 'e5m2'):
        # Exactly, as a fraction: a sum of K of them is then exact too.
        return Fraction(fp8_ref.dec(b[0], fp8_ref.FMT[fmt]))
    if fmt == 's8':
        return b[0] - 256 if b[0] >= 128 else b[0]
    if fmt == 'u8':
        return b[0]
    if fmt == 'bf16':
        return struct.unpack('<f', struct.pack('<I', struct.unpack('<H', b)[0] << 16))[0]
    return struct.unpack('<e' if fmt == 'f16' else '<f', b)[0]


def gen(pa, pb, spec='f16'):
    s = SPECS[spec]
    K, fmt, pool = s['k'], s['fmt'], _pool(s['fmt'])
    acodes = [_enc(pool[(i * 7 + 3) % len(pool)], fmt) for i in range(ROWS * K)]
    bcodes = [_enc(pool[(i * 5 + 1) % len(pool)], fmt) for i in range(K * s['cols'])]
    open(pa, 'wb').write(b''.join(acodes))
    open(pb, 'wb').write(b''.join(bcodes))


def check(pa, pb, pd, spec='f16'):
    s = SPECS[spec]
    K, w, fmt, cols = s['k'], s['w'], s['fmt'], s['cols']
    araw = open(pa, 'rb').read(ROWS * K * w)
    braw = open(pb, 'rb').read(K * cols * w)
    A = [_dec(araw[w * i:w * i + w], fmt) for i in range(ROWS * K)]
    B = [_dec(braw[w * i:w * i + w], fmt) for i in range(K * cols)]
    # An integer accumulator holds the integer, not its float value: past 2^24
    # those are different numbers, which is the whole reason the pair exists.
    iacc = s.get('iacc', False)
    D = struct.unpack(('<%di' if iacc else '<%df') % (ROWS * cols),
                      open(pd, 'rb').read(ROWS * cols * 4))
    bad = 0
    for r in range(ROWS):
        for n in range(cols):
            exact = sum(Fraction(A[r * K + k]) * Fraction(B[k * cols + n])
                        for k in range(K)) * s.get('mult', 1)
            want = int(exact) if iacc else float(exact)
            got = D[r * cols + n]
            if got != want or (iacc and exact.denominator != 1):
                if bad < 4:
                    print("  D[%d][%d] want %r got %r" % (r, n, want, got))
                bad += 1
    print("  %d of %d accumulator elements differ" % (bad, ROWS * cols))
    return bad == 0


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    spec = 'f16'
    if args and args[-1] in SPECS:
        spec = args[-1]
        args = args[:-1]
    if mode == 'gen':
        gen(args[0], args[1], spec)
    else:
        sys.exit(0 if check(args[0], args[1], args[2], spec) else 1)
