"""WEFTVOID: a game that lives on the virtual Metal device.

Two kernels in one module, from kernels/weftvoid.metal:

  voidstep   the simulation. One thread moves the ship, fires, advances bullets,
             homes the monsters in, resolves hits and spawns waves.
  voidframe  the picture. One thread per pixel traces a ray, intersects analytic
             spheres, bounces it once off the floor and shades the result.

This host does not simulate anything. It writes three numbers (move, fire, dt),
dispatches both kernels, reads back one RGBA word per pixel and one small state
buffer for the status line, and paints the pixels as half blocks. There is no
GPU: the module is plain C++ that the device runtime runs on CPU cores.

    python3 examples/vmetal/weftvoid.py build/libvmetal.dylib build/vmetal_weftvoid.dylib

Keys: a/d or left/right to move, space fires (the ship aims itself at the nearest
monster), r restarts, x quits. Without a terminal a run writes a PNG:

    python3 examples/vmetal/weftvoid.py build/libvmetal.dylib \
        build/vmetal_weftvoid.dylib --demo 240 --shot build/weftvoid.png --cols 200 --rows 60
"""
import binascii
import ctypes
import os
import select
import shutil
import struct
import sys
import time
import zlib

STATE_WORDS = 272
EB, SB, BB = 16, 112, 240
HALF = "\u2580"

def initial_state():
    s = [0.0] * STATE_WORDS
    s[2] = 0.0      # aim
    s[3] = 1.0      # health
    s[6] = 1.0      # wave
    s[7] = 0.7      # first spawn delay
    s[13] = 0.37    # spawn column seed
    for e in range(16):
        s[EB + e * 6 + 5] = 0.0
    for b in range(6):
        s[BB + b * 5 + 3] = -1.0
    return s

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

class Board:
    """The device, the two passes, and the buffers both passes share."""

    def __init__(self, lib, module, w, h):
        self.lib = lib
        self.w = w
        self.h = h
        dev = lib.vmetal_device_open(module.encode())
        if not dev:
            raise RuntimeError(lib.vmetal_last_error().decode())
        self.dev = dev
        self.ks = lib.vmetal_kernel_find(dev, b"voidstep")
        self.kf = lib.vmetal_kernel_find(dev, b"voidframe")
        if self.ks < 0 or self.kf < 0:
            raise RuntimeError(lib.vmetal_last_error().decode())
        if lib.vmetal_kernel_buffers(dev, self.ks) != 2:
            raise RuntimeError("voidstep does not declare 2 buffers")
        if lib.vmetal_kernel_buffers(dev, self.kf) != 3:
            raise RuntimeError("voidframe does not declare 3 buffers")
        self.out = lib.vmetal_new_buffer(dev, w * h * 4, None)
        self.state = lib.vmetal_new_buffer(dev, STATE_WORDS * 4, None)
        self.input = lib.vmetal_new_buffer(
            dev, 3 * 4, ctypes.byref((ctypes.c_float * 3)(0.0, 0.0, 0.0)))
        self.dims = lib.vmetal_new_buffer(
            dev, 2 * 4, ctypes.byref((ctypes.c_uint32 * 2)(w, h)))
        self.q = lib.vmetal_new_queue(dev)
        self.step_ms = 0.0
        self.frame_ms = 0.0
        self.reset()

    def put(self, buf, data):
        ctypes.memmove(self.lib.vmetal_buffer_contents(buf), data, len(data))

    def reset(self):
        s = initial_state()
        self.put(self.state, struct.pack("<%df" % STATE_WORDS, *s))

    def read_state(self):
        raw = ctypes.string_at(self.lib.vmetal_buffer_contents(self.state),
                               STATE_WORDS * 4)
        return struct.unpack("<%df" % STATE_WORDS, raw)

    def run(self, move, fire, dt):
        self.put(self.input, struct.pack("<3f", move, fire, dt))
        ms = ctypes.c_double(0.0)
        enc = self.lib.vmetal_encoder_create(self.q, self.ks)
        self.lib.vmetal_encoder_set_buffer(enc, 0, self.state, 0)
        self.lib.vmetal_encoder_set_buffer(enc, 1, self.input, 0)
        self.lib.vmetal_encoder_dispatch(enc, 1, 1)
        if self.lib.vmetal_encoder_commit(enc, ctypes.byref(ms)) != 0:
            raise RuntimeError(self.lib.vmetal_last_error().decode())
        self.lib.vmetal_encoder_release(enc)
        self.step_ms = ms.value

        enc = self.lib.vmetal_encoder_create(self.q, self.kf)
        self.lib.vmetal_encoder_set_buffer(enc, 0, self.out, 0)
        self.lib.vmetal_encoder_set_buffer(enc, 1, self.state, 0)
        self.lib.vmetal_encoder_set_buffer(enc, 2, self.dims, 0)
        self.lib.vmetal_encoder_dispatch(enc, self.w * self.h, 64)
        if self.lib.vmetal_encoder_commit(enc, ctypes.byref(ms)) != 0:
            raise RuntimeError(self.lib.vmetal_last_error().decode())
        self.lib.vmetal_encoder_release(enc)
        self.frame_ms = ms.value
        return ctypes.string_at(self.lib.vmetal_buffer_contents(self.out),
                                self.w * self.h * 4)

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
                rgba[b], rgba[b + 1], rgba[b + 2], HALF))
        out.append("".join(parts) + "\x1b[0m\n")
    return "".join(out)

def live_count(state):
    return sum(1 for e in range(16) if state[EB + e * 6 + 5] > 0.5)

def hud(state, fps, board):
    health = state[3]
    bar = int(min(1.0, max(0.0, health)) * 16)
    hp = "\u2588" * bar + "\u00b7" * (16 - bar)
    line = "HP %s  SCORE %3d  WAVE %d  MONSTERS %2d  step %.2f ms  frame %.2f ms  %3.0f fps" % (
        hp, int(state[5]), int(state[6]), live_count(state),
        board.step_ms, board.frame_ms, fps)
    if state[11] > 0.5:
        line += "   [ship lost: press r]"
    return line

def read_keys():
    try:
        data = os.read(sys.stdin.fileno(), 64)
    except OSError:
        return set()
    keys = set()
    i = 0
    while i < len(data):
        ch = data[i:i + 1]
        if ch in (b"a", b"d", b"r", b"x", b" "):
            keys.add("fire" if ch == b" " else ch.decode())
        elif ch == b"\x1b" and i + 2 < len(data) and data[i + 1:i + 2] == b"[":
            keys.add({b"C": "d", b"D": "a"}.get(data[i + 2:i + 3], "a"))
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
    cols, rows = 160, 48
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
            sys.stderr.write("weftvoid: unknown %s\n" % argv[i])
            return 2

    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if tty:
        size = shutil.get_terminal_size((cols, rows + 2))
        cols = max(40, min(cols, size.columns))
        rows = max(12, min(rows, size.lines - 2))
    w, h = cols, rows * 2

    board = Board(lib, module, w, h)
    print("weft virtual Metal: %s" % lib.vmetal_device_name().decode())
    print("kernels voidstep + voidframe  %dx%d pixels  state %d floats on the device" % (
        w, h, STATE_WORDS))

    if demo:
        # A fixed step and a scripted hand on the controls: the same run is the
        # same bytes, so the frame can be compared as a file.
        rgba = b""
        for n in range(demo):
            move = 0.0
            if n < 60:
                move = 1.0
            elif n < 120:
                move = -1.0
            fire = 1.0 if n >= 30 and (n % 80) < 45 else 0.0
            rgba = board.run(move, fire, 1.0 / 60.0)
        state = board.read_state()
        print("frames  %d  crc %08x  step %.2f ms  frame %.2f ms  hp %d  score %d  "
              "wave %d  monsters %d" % (
                  demo, binascii.crc32(rgba) & 0xFFFFFFFF, board.step_ms,
                  board.frame_ms, int(state[3] * 100), int(state[5]),
                  int(state[6]), live_count(state)))
        if shot:
            write_png(shot, w, h, rgba)
            print("png     %s  %d bytes" % (shot, os.path.getsize(shot)))
        return 0

    if not tty:
        sys.stderr.write("weftvoid: no terminal here, use --demo N [--shot FILE]\n")
        return 2

    import termios
    import tty as ttymod
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    ttymod.setcbreak(fd)
    sys.stdout.write("\x1b[?25l\x1b[2J")
    held = {}
    last = time.monotonic()
    fps = 0.0
    state = board.read_state()
    try:
        while True:
            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0.10:
                dt = 0.10
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            fresh = read_keys() if ready else set()
            if "x" in fresh:
                break
            if "r" in fresh:
                board.reset()
                held = {}
            for k in fresh:
                held[k] = now + 0.18
            move = 0.0
            fire = 0.0
            for k, until in held.items():
                if until > now:
                    if k == "a":
                        move = -1.0
                    elif k == "d":
                        move = 1.0
                    elif k == "fire":
                        fire = 1.0
            rgba = board.run(move, fire, dt)
            state = board.read_state()
            fps = 0.85 * fps + 0.15 * (1.0 / dt if dt > 0.0 else 0.0)
            sys.stdout.write(paint(rgba, w, h))
            sys.stdout.write(hud(state, fps, board) + "\x1b[K")
            sys.stdout.flush()
            left = 1.0 / 40.0 - (time.monotonic() - now)
            if left > 0.0:
                time.sleep(left)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\x1b[?25h\x1b[0m\n")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
