#include <metal_stdlib>
using namespace metal;

//     python3 examples/vmetal/weftvoid.py build/libvmetal.dylib build/vmetal_weftvoid.dylib

kernel void voidstep(device float* st [[buffer(0)]],
                     device const float* input [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    const float dt = input[2];
    if (dt <= 0.0f) return;

    const uint EB = 16u;
    const uint SB = 112u;
    const uint BB = 240u;

    float px = st[0];
    float aim = st[2];
    float health = st[3];
    float clock = st[4] + dt;
    float score = st[5];
    float wave = st[6];
    float spawn = st[7];
    float cool = st[8];
    float flash = st[10];
    const float cursor = st[9];
    float seed = st[13];

    if (st[11] > 0.5f) {
        st[4] = clock;
        flash = flash - dt * 2.0f;
        if (flash < 0.0f) flash = 0.0f;
        st[10] = flash;
        for (uint b = 0u; b < 6u; b++) {
            const uint o = BB + b * 5u;
            if (st[o + 3] >= 0.0f) {
                st[o + 3] = st[o + 3] + dt;
                if (st[o + 3] > 0.7f) st[o + 3] = -1.0f;
            }
        }
        return;
    }

    px = px + input[0] * 7.0f * dt;
    if (px < -6.3f) px = -6.3f;
    if (px > 6.3f) px = 6.3f;

    cool = cool - dt;
    if (cool < 0.0f) cool = 0.0f;

    // Fire into the first free bullet slot.
    if (input[1] > 0.5f && cool <= 0.0f) {
        uint slot = 99u;
        for (uint s = 0u; s < 16u; s++) {
            if (slot == 99u && st[SB + s * 8u + 7u] < 0.5f) slot = s;
        }
        if (slot < 99u) {
            const uint o = SB + slot * 8u;
            st[o] = px;
            st[o + 1] = 0.75f;
            st[o + 2] = 0.8f;
            st[o + 3] = sin(aim) * 26.0f;
            st[o + 4] = 0.0f;
            st[o + 5] = cos(aim) * 26.0f;
            st[o + 6] = 1.6f;
            st[o + 7] = 1.0f;
            cool = 0.12f;
        }
    }

    for (uint s = 0u; s < 16u; s++) {
        const uint o = SB + s * 8u;
        if (st[o + 7] > 0.5f) {
            st[o] = st[o] + st[o + 3] * dt;
            st[o + 1] = st[o + 1] + st[o + 4] * dt;
            st[o + 2] = st[o + 2] + st[o + 5] * dt;
            st[o + 6] = st[o + 6] - dt;
            if (st[o + 6] <= 0.0f) st[o + 7] = 0.0f;
            if (st[o + 2] > 26.0f) st[o + 7] = 0.0f;
        }
    }

    // Bullets against monsters.
    for (uint s = 0u; s < 16u; s++) {
        const uint so = SB + s * 8u;
        if (st[so + 7] > 0.5f) {
            uint done = 0u;
            for (uint e = 0u; e < 16u; e++) {
                const uint eo = EB + e * 6u;
                if (done == 0u && st[eo + 5] > 0.5f) {
                    const float dx = st[eo] - st[so];
                    const float dy = st[eo + 1] - st[so + 1];
                    const float dz = st[eo + 2] - st[so + 2];
                    const float rr = st[eo + 3] * st[eo + 3];
                    if (dx * dx + dy * dy + dz * dz < rr) {
                        done = 1u;
                        st[eo + 5] = 0.0f;
                        st[so + 7] = 0.0f;
                        score = score + 1.0f;
                        const uint bo = BB + ((uint)st[9]) * 5u;
                        st[bo] = st[eo];
                        st[bo + 1] = st[eo + 1];
                        st[bo + 2] = st[eo + 2];
                        st[bo + 3] = 0.0f;
                        st[bo + 4] = 1.0f;
                        st[9] = st[9] + 1.0f;
                        if (st[9] > 5.0f) st[9] = 0.0f;
                    }
                }
            }
        }
    }

    // Monsters close in along z and drift toward the player in x.
    uint live = 0u;
    const float speed = 3.1f + wave * 0.22f;
    for (uint e = 0u; e < 16u; e++) {
        const uint o = EB + e * 6u;
        if (st[o + 5] > 0.5f) {
            const float ez = st[o + 2] - speed * dt;
            const float ex = st[o] + (px - st[o]) * (0.85f * dt);
            st[o] = ex;
            st[o + 2] = ez;
            if (ez < 1.1f) {
                st[o + 5] = 0.0f;
                health = health - 0.25f;
                flash = 1.0f;
                const uint bo = BB + ((uint)st[9]) * 5u;
                st[bo] = px;
                st[bo + 1] = 0.75f;
                st[bo + 2] = 1.0f;
                st[bo + 3] = 0.0f;
                st[bo + 4] = 1.0f;
                st[9] = st[9] + 1.0f;
                if (st[9] > 5.0f) st[9] = 0.0f;
            } else {
                live = live + 1u;
            }
        }
    }

    // Waves arrive from the far edge at a spacing that shrinks with the wave.
    spawn = spawn - dt;
    if (spawn <= 0.0f) {
        spawn = 0.90f - wave * 0.03f;
        if (spawn < 0.34f) spawn = 0.34f;
        if (live < 10u) {
            uint slot = 99u;
            for (uint e = 0u; e < 16u; e++) {
                if (slot == 99u && st[EB + e * 6u + 5u] < 0.5f) slot = e;
            }
            if (slot < 99u) {
                seed = seed + 0.6180340f;
                if (seed > 1.0f) seed = seed - 1.0f;
                const uint o = EB + slot * 6u;
                st[o] = -6.4f + 12.8f * seed;
                st[o + 1] = 0.75f;
                st[o + 2] = 16.0f + 4.0f * seed;
                st[o + 3] = 0.62f + 0.25f * seed;
                st[o + 4] = seed;
                st[o + 5] = 1.0f;
            }
        }
    }

    float best = 1e9f;
    float ax = 0.0f;
    float az = 1.0f;
    for (uint e = 0u; e < 16u; e++) {
        const uint o = EB + e * 6u;
        if (st[o + 5] > 0.5f) {
            const float dx = st[o] - px;
            const float dz = st[o + 2];
            const float d2 = dx * dx + dz * dz;
            if (d2 < best) {
                best = d2;
                ax = dx;
                az = dz;
            }
        }
    }
    if (best < 1e8f) aim = atan2(ax, az);

    for (uint b = 0u; b < 6u; b++) {
        const uint o = BB + b * 5u;
        if (st[o + 3] >= 0.0f) {
            st[o + 3] = st[o + 3] + dt;
            if (st[o + 3] > 0.7f) st[o + 3] = -1.0f;
        }
    }

    flash = flash - dt * 2.4f;
    if (flash < 0.0f) flash = 0.0f;
    wave = 1.0f + floor(score * 0.125f);
    if (wave > 9.0f) wave = 9.0f;
    if (health <= 0.0f) {
        health = 0.0f;
        st[11] = 1.0f;
    }

    st[0] = px;
    st[2] = aim;
    st[3] = health;
    st[4] = clock;
    st[5] = score;
    st[6] = wave;
    st[7] = spawn;
    st[8] = cool;
    st[9] = cursor;
    st[10] = flash;
    st[12] = (float)live;
    st[13] = seed;
}

// The picture.
kernel void voidframe(device uint* out [[buffer(0)]],
                      device const float* st [[buffer(1)]],
                      device const uint* dims [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    const uint W = dims[0];
    const uint H = dims[1];
    if (gid >= W * H) return;
    const uint x = gid % W;
    const uint y = gid / W;

    const float px = st[0];
    const float clock = st[4];
    const float flash = st[10];
    const float over = st[11];

    const uint EB = 16u;
    const uint SB = 112u;
    const uint BB = 240u;

    const float3 cam = float3(px * 0.60f, 2.10f, -3.6f);
    const float3 tgt = float3(px * 0.25f, 0.85f, 2.0f);
    const float3 fwd = normalize(tgt - cam);
    const float3 right = normalize(cross(fwd, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = cross(right, fwd);

    const float aspect = (float)W / (float)H;
    const float cx = 2.0f * ((float)x + 0.5f) / (float)W - 1.0f;
    const float cy = 1.0f - 2.0f * ((float)y + 0.5f) / (float)H;
    const float3 rd = normalize(fwd + right * (cx * 0.85f * aspect) + up * (cy * 0.85f));

    float bestT = 1e9f;
    float3 col = float3(0.0f, 0.0f, 0.0f);
    float3 glow = float3(0.0f, 0.0f, 0.0f);
    float3 add = float3(0.0f, 0.0f, 0.0f);

    // Floor: a grid on a glossy slab.
    float floorT = 1e9f;
    if (rd.y < -0.0015f) {
        floorT = -cam.y / rd.y;
        const float3 p = cam + rd * floorT;
        const float fade = 1.0f / (1.0f + floorT * floorT * 0.006f);
        float gx = p.x * 0.7f;
        float gz = p.z * 0.7f;
        float lx = gx - floor(gx);
        float lz = gz - floor(gz);
        if (lx > 0.5f) lx = 1.0f - lx;
        if (lz > 0.5f) lz = 1.0f - lz;
        const float fx = 1.0f - lx / 0.03f;
        const float fz = 1.0f - lz / 0.03f;
        float3 fc = float3(0.010f, 0.013f, 0.024f);
        if (fx > 0.0f) fc = fc + float3(0.85f, 0.12f, 0.55f) * (fx * fx * 0.55f * fade);
        if (fz > 0.0f) fc = fc + float3(0.10f, 0.75f, 0.95f) * (fz * fz * 0.55f * fade);

        for (uint e = 0u; e < 16u; e++) {
            const uint o = EB + e * 6u;
            if (st[o + 5] > 0.5f) {
                const float dx = p.x - st[o];
                const float dz = p.z - st[o + 2];
                const float rr = st[o + 3] * 2.6f;
                const float d2 = dx * dx + dz * dz;
                if (d2 < rr * rr) {
                    const float k = d2 / (rr * rr);
                    fc = fc * (0.45f + 0.55f * k);
                }
            }
        }

        // One reflection: mirror the ray and look for an emissive sphere.
        const float3 rr = float3(rd.x, -rd.y, rd.z);
        const float3 rp = p;
        float rT = 1e9f;
        float3 rc = float3(0.0f, 0.0f, 0.0f);
        for (uint e = 0u; e < 16u; e++) {
            const uint o = EB + e * 6u;
            if (st[o + 5] > 0.5f) {
                const float3 c = float3(st[o], st[o + 1], st[o + 2]);
                const float rad = st[o + 3];
                const float3 oc = c - rp;
                const float b = dot(oc, rr);
                if (b > 0.0f) {
                    const float h2 = rad * rad - (dot(oc, oc) - b * b);
                    if (h2 > 0.0f) {
                        const float t = b - sqrt(h2);
                        if (t > 0.05f && t < rT) {
                            rT = t;
                            rc = float3(1.0f, 0.32f, 0.10f) * 0.9f;
                        }
                    }
                }
            }
        }
        for (uint s = 0u; s < 16u; s++) {
            const uint o = SB + s * 8u;
            if (st[o + 7] > 0.5f) {
                const float3 c = float3(st[o], st[o + 1], st[o + 2]);
                const float rad = 0.11f;
                const float3 oc = c - rp;
                const float b = dot(oc, rr);
                if (b > 0.0f) {
                    const float h2 = rad * rad - (dot(oc, oc) - b * b);
                    if (h2 > 0.0f) {
                        const float t = b - sqrt(h2);
                        if (t > 0.05f && t < rT) {
                            rT = t;
                            rc = float3(0.55f, 1.0f, 1.0f) * 1.4f;
                        }
                    }
                }
            }
        }
        if (rT < 1e8f) {
            float fres = 1.0f + rd.y;
            if (fres < 0.0f) fres = 0.0f;
            fc = fc + rc * (0.20f + 0.60f * fres) * fade;
        }
        col = fc;
        bestT = floorT;
    } else {
        float h = rd.y;
        if (h < 0.0f) h = 0.0f;
        col = float3(0.012f, 0.016f, 0.038f) + float3(0.02f, 0.05f, 0.12f) * h;
        const float horizon = 1.0f - h * 6.0f;
        if (horizon > 0.0f) col = col + float3(0.10f, 0.05f, 0.22f) * (horizon * horizon);
        const float hx = floor(rd.x * 260.0f);
        const float hy = floor(rd.y * 260.0f);
        const float hz = floor(rd.z * 260.0f);
        float hash = hx * 12.9898f + hy * 78.233f + hz * 37.719f;
        hash = hash - floor(hash);
        if (hash > 0.994f && rd.y > 0.05f) {
            const float star = 0.35f + 0.65f * (hash - 0.994f) / 0.006f;
            col = col + float3(0.8f, 0.85f, 1.0f) * star;
        }
    }

    {
        const float wallH = 5.0f;
        if (rd.x > 0.0002f) {
            const float t = (7.0f - cam.x) / rd.x;
            const float3 p = cam + rd * t;
            if (t > 0.05f && t < bestT && p.y > 0.0f && p.y < wallH &&
                p.z > -8.0f && p.z < 24.0f) {
                float gy = p.y - floor(p.y);
                float gz = p.z - floor(p.z);
                if (gy > 0.5f) gy = 1.0f - gy;
                if (gz > 0.5f) gz = 1.0f - gz;
                const float ly = 1.0f - gy / 0.05f;
                const float lz = 1.0f - gz / 0.05f;
                const float fade = 1.0f / (1.0f + t * t * 0.010f);
                float3 wc = float3(0.012f, 0.018f, 0.034f);
                if (ly > 0.0f) wc = wc + float3(0.10f, 0.70f, 0.95f) * (ly * 0.50f);
                if (lz > 0.0f) wc = wc + float3(0.08f, 0.42f, 0.80f) * (lz * 0.40f);
                bestT = t;
                col = wc * fade;
            }
        }
        if (rd.x < -0.0002f) {
            const float t = (-7.0f - cam.x) / rd.x;
            const float3 p = cam + rd * t;
            if (t > 0.05f && t < bestT && p.y > 0.0f && p.y < wallH &&
                p.z > -8.0f && p.z < 24.0f) {
                float gy = p.y - floor(p.y);
                float gz = p.z - floor(p.z);
                if (gy > 0.5f) gy = 1.0f - gy;
                if (gz > 0.5f) gz = 1.0f - gz;
                const float ly = 1.0f - gy / 0.05f;
                const float lz = 1.0f - gz / 0.05f;
                const float fade = 1.0f / (1.0f + t * t * 0.010f);
                float3 wc = float3(0.018f, 0.012f, 0.030f);
                if (ly > 0.0f) wc = wc + float3(0.85f, 0.16f, 0.60f) * (ly * 0.50f);
                if (lz > 0.0f) wc = wc + float3(0.70f, 0.12f, 0.45f) * (lz * 0.40f);
                bestT = t;
                col = wc * fade;
            }
        }
        if (rd.z > 0.0002f) {
            const float t = (24.0f - cam.z) / rd.z;
            const float3 p = cam + rd * t;
            if (t > 0.05f && t < bestT && p.x > -7.0f && p.x < 7.0f &&
                p.y > 0.0f && p.y < wallH) {
                const float ex2 = p.x + 7.0f;
                const float ex3 = 7.0f - p.x;
                float edge = ex2;
                if (ex3 < edge) edge = ex3;
                if (p.y < edge) edge = p.y;
                if (wallH - p.y < edge) edge = wallH - p.y;
                float3 pc = float3(0.035f, 0.020f, 0.075f);
                if (edge < 0.40f) pc = pc + float3(0.60f, 0.28f, 0.95f) * (1.0f - edge / 0.40f);
                col = pc;
                bestT = t;
            }
        }
    }

    // Monsters, bullets and the ship.
    for (uint e = 0u; e < 16u; e++) {
        const uint o = EB + e * 6u;
        if (st[o + 5] > 0.5f) {
            const float3 c = float3(st[o], st[o + 1], st[o + 2]);
            const float rad = st[o + 3];
            const float3 oc = c - cam;
            const float b = dot(oc, rd);
            const float d2 = dot(oc, oc) - b * b;
            const float near2 = d2 + rad * rad * 0.55f;
            glow = glow + float3(1.0f, 0.30f, 0.10f) * (rad * rad * 0.55f / near2);
            if (b > 0.0f) {
                const float h2 = rad * rad - d2;
                if (h2 > 0.0f) {
                    const float t = b - sqrt(h2);
                    if (t > 0.05f && t < bestT) {
                        bestT = t;
                        const float3 p = cam + rd * t;
                        const float3 n = normalize(p - c);
                        const float rim = 1.0f + dot(n, rd);
                        float3 mc = float3(0.85f, 0.10f, 0.04f) * (0.35f + 0.65f * rim);
                        mc = mc + float3(1.0f, 0.55f, 0.15f) * (rim * rim * 1.4f);
                        const float pulse = 0.85f + 0.15f * sin(clock * 4.0f + (float)e * 1.9f);
                        col = mc * pulse;
                    }
                }
            }
        }
    }

    {
        const float3 c = float3(px, 0.72f, 0.7f);
        const float rad = 0.42f;
        const float3 oc = c - cam;
        const float b = dot(oc, rd);
        const float d2 = dot(oc, oc) - b * b;
        glow = glow + float3(0.15f, 0.85f, 1.0f) * (rad * rad * 0.9f / (d2 + rad * rad * 0.6f));
        if (b > 0.0f) {
            const float h2 = rad * rad - d2;
            if (h2 > 0.0f) {
                const float t = b - sqrt(h2);
                if (t > 0.05f && t < bestT) {
                    bestT = t;
                    const float3 p = cam + rd * t;
                    const float3 n = normalize(p - c);
                    const float rim = 1.0f + dot(n, rd);
                    col = float3(0.05f, 0.45f, 0.60f) + float3(0.35f, 0.95f, 1.0f) * (rim * rim);
                }
            }
        }
    }

    for (uint s = 0u; s < 16u; s++) {
        const uint o = SB + s * 8u;
        if (st[o + 7] > 0.5f) {
            const float3 c = float3(st[o], st[o + 1], st[o + 2]);
            const float rad = 0.11f;
            const float3 oc = c - cam;
            const float b = dot(oc, rd);
            const float d2 = dot(oc, oc) - b * b;
            glow = glow + float3(0.45f, 0.95f, 1.0f) * (rad * rad * 2.4f / (d2 + rad * rad * 0.5f));
            if (b > 0.0f) {
                const float h2 = rad * rad - d2;
                if (h2 > 0.0f) {
                    const float t = b - sqrt(h2);
                    if (t > 0.05f && t < bestT) {
                        bestT = t;
                        col = float3(0.75f, 1.0f, 1.0f);
                    }
                }
            }
        }
    }

    // Booms are additive: an expanding shell plus the halo at its centre.
    for (uint bno = 0u; bno < 6u; bno++) {
        const uint o = BB + bno * 5u;
        const float age = st[o + 3];
        if (age >= 0.0f && age < 0.7f) {
            const float rad = 0.30f + age * 2.6f;
            const float3 c = float3(st[o], st[o + 1], st[o + 2]);
            const float3 oc = c - cam;
            const float b = dot(oc, rd);
            const float d2 = dot(oc, oc) - b * b;
            const float fade = 1.0f - age / 0.7f;
            add = add + float3(1.0f, 0.45f, 0.10f) * (rad * rad * 1.4f / (d2 + rad * rad) * fade);
            if (b > 0.0f) {
                const float h2 = rad * rad - d2;
                if (h2 > 0.0f) {
                    const float t = b - sqrt(h2);
                    if (t > 0.05f) {
                        add = add + float3(1.0f, 0.60f, 0.20f) * (0.55f * fade);
                    }
                }
            }
        }
    }

    col = col + add + glow * 0.055f;

    // Hit flash and the vignette.
    if (flash > 0.0f) {
        col = float3(col.x * (1.0f - 0.25f * flash) + 0.30f * flash,
                     col.y * (1.0f - 0.45f * flash),
                     col.z * (1.0f - 0.45f * flash));
    }
    if (over > 0.5f) col = col * float3(0.35f, 0.25f, 0.35f);
    const float vy = ((float)y + 0.5f) / (float)H - 0.5f;
    const float vig = 1.0f - 0.75f * (cx * cx * 0.12f + vy * vy * 0.40f);
    col = col * vig;

    // Reinhard, then a small exposure lift, then bytes.
    col = col / (float3(1.0f) + col);
    col = col * 1.25f;
    if (col.x < 0.0f) col.x = 0.0f;
    if (col.y < 0.0f) col.y = 0.0f;
    if (col.z < 0.0f) col.z = 0.0f;
    if (col.x > 1.0f) col.x = 1.0f;
    if (col.y > 1.0f) col.y = 1.0f;
    if (col.z > 1.0f) col.z = 1.0f;

    const uint r8 = (uint)(col.x * 255.0f + 0.5f);
    const uint g8 = (uint)(col.y * 255.0f + 0.5f);
    const uint b8 = (uint)(col.z * 255.0f + 0.5f);
    out[gid] = r8 | (g8 << 8) | (b8 << 16) | (255u << 24);
}
