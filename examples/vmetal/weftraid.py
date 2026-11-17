"""WEFTRAID: a first-person game drawn by a Metal kernel, played in a terminal.

The renderer is kernels/weftraid.metal. This file is the game: it holds the map,
moves the player and the monsters, and every frame it dispatches that kernel on
the virtual Metal device through the C ABI in runtime/vmetal.h, reads back one
RGBA word per pixel, and paints the result with half blocks. No GPU, no window,
no driver: the same module would run on a server, in a container, or on a phone.

    python3 examples/vmetal/weftraid.py build/libvmetal.dylib build/vmetal_weftraid.dylib

Keys: w/s forward and back, a/d strafe, q/e or left/right turn, space to shoot,
r to restart, x to quit. A frame can also be written to a PNG without a
terminal, which is how the renderer is checked:

    python3 examples/vmetal/weftraid.py build/libvmetal.dylib \
        build/vmetal_weftraid.dylib --demo 60 --shot build/weftraid.png
"""
import binascii
import ctypes
import math
import os
import select
import shutil
import struct
import sys
import time
import zlib

MAPW = 24
MAPH = 24
ENT_MAX = 16
START = (4.5, 9.5, 0.0)

# 1 and 2 and 3 are walls with their own colours, a dot is floor. The ring is a
# corridor, the four blocks are rooms with two doors each, so there is somewhere
# to run and somewhere to get cornered.
MAP_ROWS = [
    "111111111111111111111111",
    "1......................1",
    "1..22222......33333....1",
    "1..2...2......3...3....1",
    "1..2...2......3...3....1",
    "1..2...2......3...3....1",
    "1..22.22......33.33....1",
    "1......................1",
    "1......................1",
    "1..2...............2...1",
    "1..2...............2...1",
    "1..2....1111111....2...1",
    "1.......1.....1........1",
    "1.......1.....1........1",
    "1..2....1..3..1....2...1",
    "1..2....1.....1....2...1",
    "1..2....11.1111....2...1",
    "1......................1",
    "1......................1",
    "1..333.........222.....1",
    "1..3.3.........2.2.....1",
    "1..3.3.........2.2.....1",
    "1..333.........222.....1",
    "111111111111111111111111",
]

SPAWNS = [(12.5, 9.5), (20.5, 2.5), (20.5, 20.5), (2.5, 20.5), (12.5, 18.5),
          (12.5, 3.5), (6.5, 12.5), (17.5, 12.5), (12.5, 12.5),
          (8.5, 8.5), (15.5, 8.5), (8.5, 15.5), (15.5, 15.5)]

RAMP_HALF = "\u2580"

def build_map():
    cells = []
    for row in MAP_ROWS:
        for ch in row:
            cells.append(0 if ch == "." else int(ch))
    return cells

def solid(cells, x, y):
    if x < 0 or y < 0 or x >= MAPW or y >= MAPH:
        return True
    return cells[int(y) * MAPW + int(x)] != 0

class Game:
    def __init__(self, cells):
        self.cells = cells
        self.reset()

    def reset(self):
        self.px, self.py, self.ang = START
        self.health = 100.0
        self.ammo = 24
        self.score = 0
        self.wave = 1
        self.muzzle = 0.0
        self.hurt = 0.0
        self.cooldown = 0.0
        self.end = ""
        self.ents = []
        self.spawn_wave()

    def spawn_wave(self):
        count = min(ENT_MAX, 3 + self.wave)
        self.ents = []
        for i in range(count):
            x, y = SPAWNS[(i + (self.wave - 1) * 4) % len(SPAWNS)]
            self.ents.append([x + 0.0, y + 0.0, 1.0, float(self.wave % 3), 3])

    def move(self, dx, dy):
        # Axis by axis, so a wall on one side does not cancel the other.
        r = 0.22
        nx = self.px + dx
        if not solid(self.cells, nx + r, self.py) and not solid(self.cells, nx - r, self.py):
            self.px = nx
        ny = self.py + dy
        if not solid(self.cells, self.px, ny + r) and not solid(self.cells, self.px, ny - r):
            self.py = ny

    def see(self, ex, ey):
        # A grid walk from the enemy to the player: if it reaches a wall before
        # the player, the enemy does not know where the player is.
        dx = self.px - ex
        dy = self.py - ey
        steps = int(max(abs(dx), abs(dy)) * 8.0) + 1
        for i in range(1, steps):
            t = i / float(steps)
            if solid(self.cells, ex + dx * t, ey + dy * t):
                return False
        return True

    def shoot(self):
        if self.cooldown > 0.0 or self.ammo <= 0 or self.end:
            return
        self.ammo -= 1
        self.cooldown = 0.16
        self.muzzle = 1.0
        best = None
        bestd = 14.0
        for ent in self.ents:
            if ent[2] < 0.5:
                continue
            dx = ent[0] - self.px
            dy = ent[1] - self.py
            d = math.hypot(dx, dy)
            if d > bestd or d < 0.001:
                continue
            da = math.atan2(dy, dx) - self.ang
            while da > math.pi:
                da -= 2.0 * math.pi
            while da < -math.pi:
                da += 2.0 * math.pi
            if abs(da) < 0.18 and self.see(ent[0], ent[1]):
                best = ent
                bestd = d
        if best is not None:
            best[4] -= 1.0
            if best[4] <= 0.0:
                best[2] = 0.0
                self.score += 1
                if all(e[2] < 0.5 for e in self.ents):
                    self.wave += 1
                    self.ammo = min(24, self.ammo + 6)
                    self.spawn_wave()

    def update(self, dt, keys):
        if self.end:
            return
        turn = 0.0
        if "q" in keys or "left" in keys:
            turn -= 1.0
        if "e" in keys or "right" in keys:
            turn += 1.0
        self.ang += turn * 2.6 * dt

        fwd = 0.0
        side = 0.0
        if "w" in keys or "up" in keys:
            fwd += 1.0
        if "s" in keys or "down" in keys:
            fwd -= 1.0
        if "a" in keys:
            side -= 1.0
        if "d" in keys:
            side += 1.0
        if fwd != 0.0 or side != 0.0:
            n = math.hypot(fwd, side)
            speed = 3.1 * dt / n
            dx = (math.cos(self.ang) * fwd - math.sin(self.ang) * side) * speed
            dy = (math.sin(self.ang) * fwd + math.cos(self.ang) * side) * speed
            self.move(dx, dy)

        self.cooldown = max(0.0, self.cooldown - dt)
        self.muzzle = max(0.0, self.muzzle - dt * 7.0)
        self.hurt = max(0.0, self.hurt - dt * 3.6)

        for ent in self.ents:
            if ent[2] < 0.5:
                continue
            dx = self.px - ent[0]
            dy = self.py - ent[1]
            d = math.hypot(dx, dy)
            if d > 0.001 and d < 9.0 and self.see(ent[0], ent[1]):
                step = 1.5 * dt / d
                nx = ent[0] + dx * step
                ny = ent[1] + dy * step
                if not solid(self.cells, nx, ent[1]):
                    ent[0] = nx
                if not solid(self.cells, ent[0], ny):
                    ent[1] = ny
            if d < 0.85:
                self.health -= 22.0 * dt
                self.hurt = 1.0
                if self.health <= 0.0:
                    self.health = 0.0
                    self.end = "you died"

def bind(lib):
    lib.vmetal_device_name.restype = ctypes.c_char_p
    lib.vmetal_last_error.restype = ctypes.c_char_p
    lib.vmetal_device_open.restype = ctypes.c_void_p
    lib.vmetal_device_open.argtypes = [ctypes.c_char_p]
    lib.vmetal_kernel_find.restype = ctypes.c_int
    lib.vmetal_kernel_find.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.vmetal_kernel_buffers.restype = ctypes.c_int
    lib.vmetal_kernel_buffers.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.vmetal_new_buffer.restype = ctypes.c_void_p
    lib.vmetal_new_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.vmetal_buffer_contents.restype = ctypes.c_void_p
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

class Frame:
    """One device, one kernel, six buffers, reused every frame."""

    def __init__(self, lib, module, w, h):
        self.lib = lib
        self.w = w
        self.h = h
        dev = lib.vmetal_device_open(module.encode())
        if not dev:
            raise RuntimeError(lib.vmetal_last_error().decode())
        self.dev = dev
        self.k = lib.vmetal_kernel_find(dev, b"weftraid")
        if self.k < 0:
            raise RuntimeError(lib.vmetal_last_error().decode())
        nbuf = lib.vmetal_kernel_buffers(dev, self.k)
        if nbuf != 6:
            raise RuntimeError("kernel declares %d buffers, this host binds 6" % nbuf)
        cells = build_map()
        self.ents = [0.0] * (ENT_MAX * 4)
        sizes = [w * h * 4, 4 * 4, 4 * 4, 2 * 4, MAPW * MAPH * 4, ENT_MAX * 4 * 4]
        self.bufs = []
        self.bufs.append(lib.vmetal_new_buffer(dev, sizes[0], None))
        self.bufs.append(lib.vmetal_new_buffer(
            dev, sizes[1], ctypes.byref((ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0))))
        self.bufs.append(lib.vmetal_new_buffer(
            dev, sizes[2], ctypes.byref((ctypes.c_float * 4)(0.0, 0.0, 0.0, 1.0))))
        self.bufs.append(lib.vmetal_new_buffer(
            dev, sizes[3], ctypes.byref((ctypes.c_uint32 * 2)(w, h))))
        self.bufs.append(lib.vmetal_new_buffer(
            dev, sizes[4], ctypes.byref((ctypes.c_uint32 * len(cells))(*cells))))
        self.bufs.append(lib.vmetal_new_buffer(
            dev, sizes[5], ctypes.byref((ctypes.c_float * len(self.ents))(*self.ents))))
        self.q = lib.vmetal_new_queue(dev)
        self.ms = 0.0

    def put(self, index, data):
        n = len(data)
        if n:
            ctypes.memmove(self.lib.vmetal_buffer_contents(self.bufs[index]), data, n)

    def draw(self, game, clock=None):
        # Entity table: x, y, alive, kind, in the order the kernel reads it.
        for i, ent in enumerate(game.ents[:ENT_MAX]):
            base = i * 4
            self.ents[base] = ent[0]
            self.ents[base + 1] = ent[1]
            self.ents[base + 2] = ent[2]
            self.ents[base + 3] = ent[3]
        for i in range(len(game.ents), ENT_MAX):
            base = i * 4
            self.ents[base + 2] = 0.0
        cam = struct.pack("<4f", game.px, game.py, game.ang, 0.0)
        now = clock if clock is not None else (time.monotonic() - self.started)
        st = struct.pack("<4f", now, game.muzzle, game.hurt, game.health / 100.0)
        self.put(1, cam)
        self.put(2, st)
        self.put(5, struct.pack("<%df" % len(self.ents), *self.ents))

        enc = self.lib.vmetal_encoder_create(self.q, self.k)
        for i in range(6):
            self.lib.vmetal_encoder_set_buffer(enc, i, self.bufs[i], 0)
        self.lib.vmetal_encoder_dispatch(enc, self.w * self.h, 64)
        ms = ctypes.c_double(0.0)
        rc = self.lib.vmetal_encoder_commit(enc, ctypes.byref(ms))
        self.lib.vmetal_encoder_release(enc)
        if rc != 0:
            raise RuntimeError(self.lib.vmetal_last_error().decode())
        self.ms = ms.value
        n = self.w * self.h * 4
        return ctypes.string_at(self.lib.vmetal_buffer_contents(self.bufs[0]), n)

def write_png(path, w, h, rgba):
    rgb = bytearray(w * h * 3)
    for i in range(w * h):
        rgb[i * 3] = rgba[i * 4]
        rgb[i * 3 + 1] = rgba[i * 4 + 1]
        rgb[i * 3 + 2] = rgba[i * 4 + 2]
    raw = b"".join(b"\x00" + bytes(rgb[y * w * 3:(y + 1) * w * 3]) for y in range(h))

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", binascii.crc32(tag + body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))

def paint(rgba, w, h):
    """Half blocks: the upper pixel is the foreground, the lower one the back."""
    out = ["\x1b[H"]
    for row in range(h // 2):
        parts = []
        top = row * 2 * w * 4
        bot = top + w * 4
        for cx in range(w):
            t = top + cx * 4
            b = bot + cx * 4
            parts.append("\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm%s" % (
                rgba[t], rgba[t + 1], rgba[t + 2],
                rgba[b], rgba[b + 1], rgba[b + 2], RAMP_HALF))
        out.append("".join(parts) + "\x1b[0m\n")
    return "".join(out)

def hud(game, fps, ms):
    bar = int(min(1.0, max(0.0, game.health / 100.0)) * 20)
    hp = "\u2588" * bar + "\u00b7" * (20 - bar)
    line = "HP %s %3d  AMMO %2d  SCORE %2d  WAVE %d  %4.1f ms  %2.0f fps" % (
        hp, int(game.health), game.ammo, game.score, game.wave, ms, fps)
    if game.end:
        line += "   [%s: press r]" % game.end
    return line

def read_keys(raw_ready):
    keys = set()
    try:
        data = os.read(sys.stdin.fileno(), 64)
    except OSError:
        return keys
    i = 0
    while i < len(data):
        ch = data[i:i + 1]
        if ch in (b"w", b"s", b"a", b"d", b"q", b"e", b"r", b"x", b" "):
            keys.add("shoot" if ch == b" " else ch.decode())
        elif ch == b"\x1b" and i + 2 < len(data) and data[i + 1:i + 2] == b"[":
            arrow = data[i + 2:i + 3]
            keys.add({b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(
                arrow, "up"))
            i += 2
        elif ch == b"\x03":
            keys.add("x")
        i += 1
    return keys

def main(argv):
    if len(argv) < 3:
        sys.stderr.write("usage: %s <libvmetal> <module> [--demo N] [--shot FILE] "
                         "[--cols N] [--rows N]\n" % argv[0])
        return 2
    lib = bind(ctypes.CDLL(argv[1]))
    module = argv[2]
    demo = 0
    shot = None
    cols, rows = 100, 30
    i = 3
    while i < len(argv):
        if argv[i] == "--demo":
            demo = int(argv[i + 1])
            i += 2
        elif argv[i] == "--shot":
            shot = argv[i + 1]
            i += 2
        elif argv[i] == "--cols":
            cols = int(argv[i + 1])
            i += 2
        elif argv[i] == "--rows":
            rows = int(argv[i + 1])
            i += 2
        else:
            sys.stderr.write("weftraid: unknown %s\n" % argv[i])
            return 2

    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if tty:
        size = shutil.get_terminal_size((cols, rows + 2))
        cols = max(40, min(cols, size.columns))
        rows = max(12, min(rows, size.lines - 2))
    w, h = cols, rows * 2

    game = Game(build_map())
    frame = Frame(lib, module, w, h)
    frame.started = time.monotonic()

    print("weft virtual Metal: %s" % lib.vmetal_device_name().decode())
    print("kernel weftraid  %dx%d pixels  map %dx%d  entities %d" % (
        w, h, MAPW, MAPH, len(game.ents)))

    if demo:
        # No terminal and no clock in the simulation: a fixed step, a scripted
        # trigger, so the same run gives the same bytes and can be compared.
        fps = 0.0
        digest = ""
        for n in range(demo):
            # The shader reads a clock for the weapon bob and the low-health
            # pulse, so the demo hands it the frame number: a run is the same
            # bytes twice, and the frame can be compared as a file.
            nxt = {"w": True} if n < 15 else {}
            if 15 <= n < 19:
                nxt = {"q": True}
            game.update(1.0 / 30.0, nxt)
            if n == 25 or n == 45:
                game.shoot()
            rgba = frame.draw(game, float(n) / 30.0)
            digest = "%08x" % (binascii.crc32(rgba) & 0xFFFFFFFF)
            fps = 1.0 / (frame.ms / 1000.0) if frame.ms > 0.0 else 0.0
        print("frames  %d  last crc %s  %.3f ms  %.0f fps  hp %d  score %d" % (
            demo, digest, frame.ms, fps, int(game.health), game.score))
        if shot:
            write_png(shot, w, h, rgba)
            print("png     %s  %d bytes" % (shot, os.path.getsize(shot)))
        return 0

    if not tty:
        sys.stderr.write("weftraid: no terminal here, use --demo N [--shot FILE]\n")
        return 2

    import termios
    import tty as ttymod
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    ttymod.setcbreak(fd)
    sys.stdout.write("\x1b[?25l\x1b[2J")
    held = {}
    last = time.monotonic()
    fps = 0.0
    try:
        while True:
            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0.10:
                dt = 0.10
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            fresh = read_keys(ready) if ready else set()
            if "x" in fresh:
                break
            if "r" in fresh and game.end:
                game.reset()
                held = {}
            if "shoot" in fresh:
                game.shoot()
            # A key is held for a moment after it was read, so key repeat feels
            # like movement and a single tap still moves one step.
            for k in fresh:
                if k != "shoot":
                    held[k] = now + 0.22
            keys = set(k for k, until in held.items() if until > now)
            game.update(dt, keys)
            rgba = frame.draw(game)
            fps = 0.85 * fps + 0.15 * (1.0 / dt if dt > 0.0 else 0.0)
            sys.stdout.write(paint(rgba, w, h))
            sys.stdout.write(hud(game, fps, frame.ms) + "\x1b[K")
            sys.stdout.flush()
            left = 1.0 / 30.0 - (time.monotonic() - now)
            if left > 0.0:
                time.sleep(left)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\x1b[?25h\x1b[0m\n")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
