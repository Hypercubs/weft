#include <metal_stdlib>
using namespace metal;

//     python3 examples/vmetal/weftraid.py build/libvmetal.dylib \
kernel void weftraid(device uint* out [[buffer(0)]],
                     device const float* cam [[buffer(1)]],
                     device const float* state [[buffer(2)]],
                     device const uint* dims [[buffer(3)]],
                     device const uint* map [[buffer(4)]],
                     device const float* ents [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    const uint W = dims[0];
    const uint H = dims[1];
    if (gid >= W * H) return;

    const uint x = gid % W;
    const uint y = gid / W;

    const float px = cam[0];
    const float py = cam[1];
    const float ang = cam[2];
    const float pitch = cam[3];

    const float now = state[0];
    const float muzzle = state[1];
    const float hurt = state[2];
    const float hp01 = state[3];

    // The camera plane.
    const float dirX = cos(ang);
    const float dirY = sin(ang);
    const float planeX = -dirY * 1.10f;
    const float planeY = dirX * 1.10f;

    const float camX = 2.0f * ((float)x + 0.5f) / (float)W - 1.0f;
    const float rayX = dirX + planeX * camX;
    const float rayY = dirY + planeY * camX;

    // Grid walk.
    int mx = (int)floor(px);
    int my = (int)floor(py);
    if (mx < 0) mx = 0;
    if (my < 0) my = 0;
    if (mx > 23) mx = 23;
    if (my > 23) my = 23;

    float ddX = 1.0f / rayX;
    float ddY = 1.0f / rayY;
    if (ddX < 0.0f) ddX = -ddX;
    if (ddY < 0.0f) ddY = -ddY;

    int stepX = 1;
    int stepY = 1;
    if (rayX < 0.0f) stepX = -1;
    if (rayY < 0.0f) stepY = -1;

    float sdX = ((float)mx + 1.0f - px) * ddX;
    float sdY = ((float)my + 1.0f - py) * ddY;
    if (rayX < 0.0f) sdX = (px - (float)mx) * ddX;
    if (rayY < 0.0f) sdY = (py - (float)my) * ddY;

    uint hit = 0u;
    uint cell = 0u;
    uint side = 0u;
    float dist = 40.0f;

    for (uint s = 0u; s < 64u; s++) {
        if (hit == 0u) {
            if (sdX < sdY) {
                mx = mx + stepX;
                dist = sdX;
                sdX = sdX + ddX;
                side = 0u;
            } else {
                my = my + stepY;
                dist = sdY;
                sdY = sdY + ddY;
                side = 1u;
            }
            if (mx < 0 || mx > 23 || my < 0 || my > 23) {
                hit = 1u;
                cell = 7u;
            } else {
                cell = map[my * 24 + mx];
                if (cell > 0u) hit = 1u;
            }
        }
    }

    const float lineH = (float)H / (dist + 0.0001f);
    const float center = (float)H * 0.5f + pitch * (float)H;
    const float top = center - lineH * 0.5f;
    const float bottom = center + lineH * 0.5f;

    // Wall id to colour. Three wall kinds is enough to read distance and turns.
    float wr = 0.50f;
    float wg = 0.40f;
    float wb = 0.32f;
    if (cell == 2u) {
        wr = 0.34f;
        wg = 0.38f;
        wb = 0.44f;
    } else if (cell == 3u) {
        wr = 0.28f;
        wg = 0.44f;
        wb = 0.30f;
    } else if (cell == 7u) {
        wr = 0.06f;
        wg = 0.07f;
        wb = 0.12f;
    }

    float pr = 0.0f;
    float pg = 0.0f;
    float pb = 0.0f;

    const float fy = (float)y + 0.5f;
    if (fy >= top && fy <= bottom) {
        float wallX = px + dist * rayX;
        if (side == 0u) wallX = py + dist * rayY;
        wallX = wallX - floor(wallX);

        const float v = (fy - top) / (lineH + 0.0001f);
        const float rows = 10.0f;
        const float ry = v * rows;
        const float rowi = floor(ry);
        const float fx = wallX * 5.0f + rowi * 0.5f;
        const float bx = fx - floor(fx);

        float shade = 1.0f;
        if (bx < 0.05f || (ry - rowi) < 0.09f) shade = 0.60f;
        float fogW = 1.0f / (1.0f + dist * dist * 0.030f);
        if (side == 1u) fogW = fogW * 0.70f;
        shade = shade * fogW;
        pr = wr * shade;
        pg = wg * shade;
        pb = wb * shade;
    } else if (fy < top) {
        const float fd = (0.5f * (float)H) / (center - fy + 0.0001f);
        const float wx = px + rayX * fd;
        const float wy = py + rayY * fd;
        const float gx2 = wx * 0.5f;
        const float gy2 = wy * 0.5f;
        const float lx = gx2 - floor(gx2) - 0.5f;
        const float ly = gy2 - floor(gy2) - 0.5f;
        const float near = lx * lx + ly * ly;
        float lamp = 0.045f;
        if (near < 0.055f) lamp = 1.0f - near * 12.0f;
        const float fogC = 1.0f / (1.0f + fd * fd * 0.012f);
        pr = (0.05f + 0.72f * lamp) * fogC;
        pg = (0.06f + 0.70f * lamp) * fogC;
        pb = (0.11f + 0.60f * lamp) * fogC;
    } else {
        const float fd = (0.5f * (float)H) / (fy - center + 0.0001f);
        const float wx = px + rayX * fd;
        const float wy = py + rayY * fd;
        const float cx2 = floor(wx);
        const float cy2 = floor(wy);
        const float par = (cx2 + cy2) - 2.0f * floor((cx2 + cy2) * 0.5f);
        float tone = 0.20f;
        if (par > 0.5f) tone = 0.13f;
        const float ex = wx - cx2;
        const float ey = wy - cy2;
        if (ex < 0.04f || ey < 0.04f) tone = tone * 0.55f;
        const float fogF = 1.0f / (1.0f + fd * fd * 0.020f);
        pr = tone * 0.95f * fogF;
        pg = tone * 1.00f * fogF;
        pb = tone * 0.80f * fogF;
    }

    // Entities.
    const float invDet = 1.0f / (planeX * dirY - dirX * planeY);
    for (uint e = 0u; e < 16u; e++) {
        const float alive = ents[e * 4u + 2u];
        if (alive > 0.5f) {
            const float relX = ents[e * 4u] - px;
            const float relY = ents[e * 4u + 1u] - py;
            const float tx = invDet * (dirY * relX - dirX * relY);
            const float ty = invDet * (planeX * relY - planeY * relX);
            if (ty > 0.25f && ty < dist) {
                const float sx = 0.5f * (float)W * (1.0f + tx / ty);
                const float sh = (float)H / ty;
                const float nx = ((float)x + 0.5f - sx) / (sh * 0.30f);
                const float ny = (fy - center) / (sh * 0.55f) + 0.25f;
                const float rr = nx * nx + ny * ny;
                if (rr < 1.0f) {
                    const float kind = ents[e * 4u + 3u];
                    float sr = 0.34f;
                    float sg = 0.55f;
                    float sb = 0.30f;
                    if (kind > 0.5f && kind < 1.5f) {
                        sr = 0.66f;
                        sg = 0.44f;
                        sb = 0.24f;
                    } else if (kind >= 1.5f) {
                        sr = 0.62f;
                        sg = 0.22f;
                        sb = 0.22f;
                    }
                    const float edge = 0.70f + 0.30f * (1.0f - rr);
                    sr = sr * edge;
                    sg = sg * edge;
                    sb = sb * edge;
                    const float ex2 = (nx - 0.20f) * (nx - 0.20f) + (ny + 0.42f) * (ny + 0.42f);
                    const float ex3 = (nx + 0.20f) * (nx + 0.20f) + (ny + 0.42f) * (ny + 0.42f);
                    if (ex2 < 0.018f || ex3 < 0.018f) {
                        sr = 0.95f;
                        sg = 0.92f;
                        sb = 0.35f;
                    } else if (ny > -0.02f && ny < 0.10f && nx > -0.24f && nx < 0.24f) {
                        sr = sr * 0.30f;
                        sg = sg * 0.25f;
                        sb = sb * 0.25f;
                    }
                    const float fogE = 1.0f / (1.0f + ty * ty * 0.030f);
                    pr = sr * fogE;
                    pg = sg * fogE;
                    pb = sb * fogE;
                }
            }
        }
    }

    // The weapon.
    const float gx = (((float)x + 0.5f) - 0.5f * (float)W) / (float)H;
    const float gy = ((float)y + 0.5f) / (float)H + muzzle * 0.035f;
    if (gx > -0.070f && gx < 0.070f && gy > 0.54f && gy < 1.0f) {
        const float hl = (gx + 0.070f) / 0.140f;
        pr = 0.09f + 0.18f * hl;
        pg = 0.09f + 0.18f * hl;
        pb = 0.11f + 0.22f * hl;
        if (gy < 0.70f) {
            const float band = (gy - 0.54f) * 48.0f;
            if (band - floor(band) < 0.32f) {
                pr = pr * 0.5f;
                pg = pg * 0.5f;
                pb = pb * 0.55f;
            }
        }
    }
    const float hand = gy - 0.70f;
    if (hand > 0.0f && hand < 0.32f) {
        const float half = 0.10f + hand * 0.55f;
        const float wob = 0.02f * sin(now * 1.7f + gx * 9.0f);
        if (gx > -half && gx < half) {
            pr = 0.36f + wob;
            pg = 0.26f + wob;
            pb = 0.19f + wob;
        }
    }
    const float fdy = gy - 0.532f;
    const float flashR = gx * gx + fdy * fdy * 0.45f;
    float flash = 0.0f;
    if (muzzle > 0.02f && flashR < 0.030f * muzzle) {
        flash = 1.0f - flashR / (0.030f * muzzle);
    }
    if (muzzle > 0.02f && gx > -0.016f && gx < 0.016f && fdy > -0.17f && fdy < 0.02f) {
        flash = 0.85f;
    }
    if (flash > 0.0f) {
        pr = pr * (1.0f - flash) + 1.00f * flash;
        pg = pg * (1.0f - flash) + 0.94f * flash;
        pb = pb * (1.0f - flash) + 0.52f * flash;
    }

    if (hurt > 0.0f) {
        pr = pr * (1.0f - 0.30f * hurt) + 0.34f * hurt;
        pg = pg * (1.0f - 0.45f * hurt);
        pb = pb * (1.0f - 0.45f * hurt);
    }
    if (hp01 < 0.30f) {
        const float pulse = 0.5f + 0.5f * sin(now * 7.0f);
        const float k = (0.30f - hp01) * 1.7f * pulse;
        if (k > 0.0f) {
            pr = pr * (1.0f - 0.45f * k) + 0.35f * k;
            pg = pg * (1.0f - 0.55f * k);
            pb = pb * (1.0f - 0.55f * k);
        }
    }
    const float vy = ((float)y + 0.5f) / (float)H - 0.5f;
    const float vig = 1.0f - 0.85f * (camX * camX * 0.16f + vy * vy * 0.50f);
    pr = pr * vig;
    pg = pg * vig;
    pb = pb * vig;

    if (pr < 0.0f) pr = 0.0f;
    if (pg < 0.0f) pg = 0.0f;
    if (pb < 0.0f) pb = 0.0f;
    if (pr > 1.0f) pr = 1.0f;
    if (pg > 1.0f) pg = 1.0f;
    if (pb > 1.0f) pb = 1.0f;

    const uint r8 = (uint)(pr * 255.0f + 0.5f);
    const uint g8 = (uint)(pg * 255.0f + 0.5f);
    const uint b8 = (uint)(pb * 255.0f + 0.5f);
    out[gid] = r8 | (g8 << 8) | (b8 << 16) | (255u << 24);
}
