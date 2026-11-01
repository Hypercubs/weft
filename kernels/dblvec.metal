#include <metal_stdlib>
using namespace metal;

// `double` is not supported on Apple GPUs — the Metal compiler rejects it — so
// the double vectors are checked on the CPU only, against the same arithmetic
// written out with scalars. What that covers is the operators and the layout,
// which is all this front end can get wrong.
kernel void dblvec(device const float* in [[buffer(0)]],
                   device float* out      [[buffer(1)]],
                   uint gid               [[thread_position_in_grid]]) {
    double a = (double)in[gid];
    double2 d2 = double2(a, 2.0);
    double2 e2 = d2 * 3.0 + double2(1.0, 0.5);
    bool2 ok2 = bool2{e2.x == a * 3.0 + 1.0, e2.y == 2.0 * 3.0 + 0.5};

    double3 d3 = double3(a, -a, 0.25);
    double3 e3 = d3 * d3 - double3(1.0, 1.0, 2.0);
    bool3 ok3 = bool3{e3.x == a * a - 1.0, e3.y == (-a) * (-a) - 1.0, e3.z == 0.0625 - 2.0};

    double4 d4 = double4(1.0, 2.0, 3.0, 4.0);
    bool4 ok4 = (d4 * 0.5) == double4(0.5, 1.0, 1.5, 2.0);

    bool ok = all(ok2) && ok3.x && ok3.y && ok3.z && all(ok4);
    out[gid] = ok ? 1.0f : 0.0f;
}
