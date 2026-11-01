#!/usr/bin/env python3
"""Find the statements in an emitted Metal kernel that kill this host's compiler.

The Metal compiler is two stages, and they fail differently. `newLibraryWithSource`
builds a library from source; `newComputePipelineStateWithFunction` then generates
machine code. A shader can pass the first and take the second down hard — the
diagnostic is `XPC_ERROR_CONNECTION_INTERRUPTED`, which names the transport rather
than the construct, so it says nothing about what is wrong.

This searches for the construct. It removes statements from the body, one group at
a time, and asks the compiler after each removal:

  crash  the run still fails, so what is left is still enough to reproduce it
  ok     the pipeline builds, so this group is part of what it chokes on
  error  the removal does not compile, so it cannot be removed (a declaration
         that something later names, say)

`crash` shrinks the candidate set, `ok` and `error` grow the granularity, and what
survives is a minimal set of lines whose presence reproduces the failure. That is
usually one statement, and it is usually a type or an operation the back end has no
lowering for — on this host it was a `__int128` multiply that a 32-bit limb product
replaced.

    ./build/weft run file.ptx --emit-gpu > out.metal   # the source, nothing else
    python3 tools/msl_bisect.py out.metal
    python3 tools/msl_bisect.py out.metal --timeout 60

The checker is `tools/msl_check.mm` — the same compiler the launcher hands its
source to — so a failure here is a failure there, with no GPU in the loop.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER_SRC = os.path.join(ROOT, "tools", "msl_check.mm")
CHECKER = os.path.join(tempfile.gettempdir(), "weft-mslcheck")


def build_checker():
    """Compile the checker the first time, and reuse it after that."""
    if os.path.exists(CHECKER) and os.path.getmtime(CHECKER) > os.path.getmtime(CHECKER_SRC):
        return True
    cxx = "/usr/bin/clang++" if os.path.exists("/usr/bin/clang++") else "clang++"
    cmd = [cxx, "-fobjc-arc", "-std=c++17", "-O1", "-framework", "Metal",
           "-framework", "Foundation", CHECKER_SRC, "-o", CHECKER]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", "replace"))
        return False
    return True


def classify(text, timeout):
    """Compile one candidate. `crash` if the compiler dies, `error` if it refuses."""
    with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        r = subprocess.run([CHECKER, path, "--pipeline"], capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "crash", "timed out after %ds" % timeout
    finally:
        os.unlink(path)
    err = (r.stderr + r.stdout).decode("utf-8", "replace").strip()
    if r.returncode == 0:
        return "ok", ""
    # A dead compiler service and a diagnostic are different answers. The message
    # for the first names the transport; the second names the source line.
    if "XPC_ERROR" in err or "interrupted connection" in err or not re.search(r"error:", err):
        return "crash", err.splitlines()[0] if err else "no output"
    return "error", err.splitlines()[0]


def statements(lines):
    """The lines that can be removed on their own: statements, not labels or braces.

    Only the last kernel's body counts. The helpers above it are shared by every
    kernel in the file, and a statement removed from one of them changes the file
    rather than the kernel — which reads as a cure when it is really a warning.
    """
    start = 0
    for i, line in enumerate(lines):
        if line.startswith(("kernel ", "fragment ", "vertex ")):
            start = i
    out = []
    for i, line in enumerate(lines):
        if i <= start:
            continue
        s = line.strip()
        if not s.endswith(";"):
            continue
        if s in ("break;", "continue;"):
            continue
        if s.startswith(("case ", "default:", "return;", "#", "//")):
            continue
        if not line.startswith("  "):  # a top-level definition, not a body statement
            continue
        out.append(i)
    return out


def with_removed(lines, drop):
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def shrink(lines, cand, timeout):
    """Standard ddmin over the removable statements, with a third outcome."""
    keep = list(cand)
    gran = 2
    while len(keep) >= 2 and gran <= len(keep):
        size = max(1, len(keep) // gran)
        chunks = [keep[i:i + size] for i in range(0, len(keep), size)]
        removed = False
        for ch in chunks:
            trial = [i for i in keep if i not in set(ch)]
            what, _ = classify(with_removed(lines, set(trial) | (set(range(len(lines))) - set(cand))), timeout)
            # Removing this chunk keeps the crash alive, so the cause is elsewhere.
            if what == "crash":
                keep = trial
                gran = max(2, gran - 1)
                removed = True
                break
        if not removed:
            if gran == len(keep):
                break
            gran = min(len(keep), gran * 2)
    return keep


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    path = sys.argv[1]
    timeout = 30
    if "--timeout" in sys.argv:
        timeout = int(sys.argv[sys.argv.index("--timeout") + 1])

    if not build_checker():
        return 2
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")

    what, msg = classify(text, timeout)
    if what == "ok":
        print("%s builds and takes a pipeline — nothing to bisect" % path)
        return 0
    print("%s: %s" % (what, msg))
    if what != "crash":
        return 1

    cand = statements(lines)
    if not cand:
        print("no removable statements found")
        return 1
    print("%d candidate statements; bisecting…" % len(cand))
    keep = shrink(lines, cand, timeout)

    # Two different answers are worth separating, because they point at different
    # things. Either a small set of statements reproduces the failure on its own —
    # that is the construct, and the search above has already found it — or the body
    # fails as a whole, in which case there is no construct and the question becomes
    # which statements matter.
    # Remove the candidates the search let go of, and leave every line that is not a
    # candidate alone: the kernel signature and the helpers stay, or nothing compiles.
    rest = set(cand) - set(keep)
    alone, _ = classify(with_removed(lines, rest), timeout)
    if alone == "crash" and len(keep) <= 6:
        print("\nthe smallest reproducing set is %d statement(s):" % len(keep))
        for i in keep:
            print("  %4d  %s" % (i + 1, lines[i].strip()))
        return 0

    print("\nno set of %d statements reproduces it alone: the body fails as a whole,"
          % len(keep))
    print("so the cause is not one construct but something the whole body crosses.")
    print("Removing each statement on its own to see which ones matter:")
    cures = []
    for i in cand:
        what, _ = classify(with_removed(lines, {i}), timeout)
        if what == "ok":
            cures.append(i)
    print("  %d of %d statements, removed one at a time, make the pipeline build:"
          % (len(cures), len(cand)))
    for i in cures[:10]:
        print("  %4d  %s" % (i + 1, lines[i].strip()))
    if len(cures) > 10:
        print("  … and %d more" % (len(cures) - 10))
    if not cures:
        print("  (none: this one needs a combination, and the group above is the")
        print("   smallest the search found)")
    else:
        print("  Many single removals curing it means a limit the body crosses — a")
        print("  register count, an instruction count — rather than one unsupported")
        print("  construct. Look for the widest type or operation in it: a 128-bit")
        print("  multiply in a kernel this size is exactly what this looked like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
