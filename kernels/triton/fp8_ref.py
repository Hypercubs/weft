"""An fp8 reference written from the formats' definitions, not from the runtime.

e4m3 (the OCP "FN" variant `cvt.rn.satfinite.e4m3x2.f32` produces): 1 sign bit,
4 exponent bits biased 7, 3 mantissa bits, no infinities, and the largest finite
magnitude is 448. e5m2: 1 sign bit, 5 exponent bits biased 15, 2 mantissa bits,
infinities and NaN at the top exponent, largest finite magnitude 57344.

In both, subnormals carry the mantissa at a fixed quantum (2^-9 and 2^-16), so
the smallest positive value is one quantum. Rounding is to nearest, ties to even,
done in exact integer arithmetic so a comparison cannot hide a rounding
difference.
"""
import sys, struct
from fractions import Fraction

class Fmt:
    def __init__(self, ebits, mbits, inf_nan):
        self.ebits, self.mbits, self.inf_nan = ebits, mbits, inf_nan
        self.bias = (1 << (ebits - 1)) - 1
        self.maxexp = (1 << ebits) - 1
        self.emax = self.maxexp - (1 if inf_nan else 0)  # largest finite exponent
        # The subnormal quantum is one binade below the smallest normal, so its
        # exponent is bias+mbits-1, not bias+mbits.
        self.quantum = Fraction(1, 1 << (self.bias + mbits - 1))
        # e4m3 has no infinities, and its top mantissa at the top exponent is
        # NaN: the largest finite code is one below it, not all ones.
        self.maxmag = (self.emax << mbits) | ((1 << mbits) - (1 if inf_nan else 2))

E4M3 = Fmt(4, 3, False)
E5M2 = Fmt(5, 2, True)
FMT = {'e4m3': E4M3, 'e5m2': E5M2}

def rne(v):
    f = Fraction(v)
    n = int(f)
    r = f - n
    if r > Fraction(1, 2):
        n += 1
    elif r == Fraction(1, 2) and n % 2:
        n += 1
    return n

def _frexp(ax):
    # ax = m * 2^e with m in [0.5, 1), in exact arithmetic
    f = Fraction(ax)
    e = 0
    while f >= 1:
        f /= 2
        e += 1
    while f < Fraction(1, 2):
        f *= 2
        e -= 1
    return float(f), e

def enc(x, f):
    if x != x:
        return 0x7F
    sign = 0x80 if (x < 0 or (x == 0 and struct.pack('<f', x)[3] & 0x80)) else 0
    ax = abs(x)
    maxval = float(dec(f.maxmag, f))
    if ax > maxval:
        return sign | f.maxmag
    if ax == 0.0:
        return sign
    m, e = _frexp(ax)
    E = e - 1 + f.bias
    if E <= 0:
        q = rne(ax / float(f.quantum))
        return sign | ((1 << f.mbits) if q >= (1 << f.mbits) else q)
    if E >= f.emax + 1:
        return sign | f.maxmag
    q = rne((m * 2 - 1) * (1 << f.mbits))
    if q == (1 << f.mbits):
        q = 0
        E += 1
    if E >= f.emax + 1:
        return sign | f.maxmag
    out = (E << f.mbits) + q
    return sign | (f.maxmag if out > f.maxmag else out)

def dec(b, f):
    sign = -1.0 if b & 0x80 else 1.0
    E = (b >> f.mbits) & f.maxexp
    mag = b & ((1 << f.mbits) - 1)
    if E == 0:
        d = Fraction(mag) * f.quantum
    else:
        d = Fraction((1 << f.mbits) + mag) * Fraction(2) ** (E - f.bias - f.mbits)
    return sign * float(d)

def gen(na, nb, path_a, path_b, f):
    # Magnitudes that are exact binary fractions in both formats, so every
    # product and every sum of 32 of them is exactly representable: a difference
    # is a layout error, not a rounding one.
    pool = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, -0.5, -1.0, -2.0, 0.25, 0.0, 2.5]
    codes = [enc(v, f) for v in pool]
    a = bytes(codes[(i * 7 + 3) % len(codes)] for i in range(na))
    b = bytes(codes[(i * 5 + 1) % len(codes)] for i in range(nb))
    open(path_a, 'wb').write(a)
    open(path_b, 'wb').write(b)

def gen_cvt(path, f):
    # Values that exercise every branch: saturation (449, 500), the largest
    # finite value, a subnormal, a tie that rounds to zero, ties-to-even in both
    # directions, negatives, and mantissas that round up into the exponent
    # (500, -7.75).
    vals = [0.0, 1.0, -1.0, 0.5, 448.0, 449.0, 500.0, 0.001953125, 0.0009765625,
            1.0625, 1.09375, 2.5, 3.0, -0.25, 0.125, 100.0, 0.3, -7.75]
    if f is E5M2:
        vals = [0.0, 1.0, -1.0, 0.5, 57344.0, 57345.0, 60000.0, 1.52587890625e-05,
                7.62939453125e-06, 1.0625, 1.09375, 2.5, 3.0, -0.25, 0.125, 100.0,
                0.3, -7.75]
    xs = (vals * 8)[:128]
    open(path, 'wb').write(struct.pack('<128f', *xs))

def check_matmul(pa, pb, pd, f):
    A = open(pa, 'rb').read()
    B = open(pb, 'rb').read()
    D = struct.unpack('<128f', open(pd, 'rb').read(512))
    bad = 0
    for r in range(16):
        for n in range(8):
            want = sum(Fraction(dec(A[r * 32 + k], f)) * Fraction(dec(B[k * 8 + n], f))
                       for k in range(32))
            got = D[r * 8 + n]
            if got != float(want):
                if bad < 4:
                    print("  D[%d][%d] want %r got %r" % (r, n, float(want), got))
                bad += 1
    print("  %d of 128 tile elements differ" % bad)
    return bad == 0

def check_cvt(pin, pout, f):
    xs = struct.unpack('<128f', open(pin, 'rb').read(512))
    out = open(pout, 'rb').read(128)
    bad = 0
    for i, x in enumerate(xs):
        want = enc(x, f)
        got = out[i]
        if got != want:
            if bad < 4:
                print("  cvt[%d] %r want 0x%02x got 0x%02x" % (i, x, want, got))
            bad += 1
    print("  %d of 128 converted bytes differ" % bad)
    return bad == 0

def fmt_of(args, default='e4m3'):
    return FMT[args[-1]] if args and args[-1] in FMT else FMT[default]

if __name__ == '__main__':
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    f = fmt_of(rest)
    if cmd == 'gen':
        gen(int(rest[0]), int(rest[1]), rest[2], rest[3], f)
    elif cmd == 'matmul':
        sys.exit(0 if check_matmul(rest[0], rest[1], rest[2], f) else 1)
    elif cmd == 'gen-cvt':
        gen_cvt(rest[0], f)
    elif cmd == 'cvt':
        sys.exit(0 if check_cvt(rest[0], rest[1], f) else 1)
