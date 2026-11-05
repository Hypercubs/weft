# The launcher is one file on every platform. On macOS its GPU column needs
# Metal + Foundation and it is an Objective-C++ file; elsewhere there is no GPU
# backend, the Metal-only parts are compiled out, and it is plain C++ that links
# against nothing but libdl (nothing at all on Windows).
UNAME := $(shell uname -s)
EXE     :=
ifeq ($(UNAME),Darwin)
# Apple's clang shim by full path, not whatever `clang++` a Homebrew LLVM put
# first on PATH: that one cannot link the ObjC message sends this uses against
# Apple's frameworks, and the failure reads as a missing SDK.
CXX      := $(shell test -x /usr/bin/clang++ && echo /usr/bin/clang++ || echo clang++)
CXXFLAGS := -std=c++20 -O2 -fobjc-arc -I runtime -I include
FW       := -framework Metal -framework Foundation
MAIN     := src/main.mm
else ifneq (,$(strip $(filter MINGW% MSYS% CYGWIN%,$(UNAME))))
# Windows under MSYS2: no libdl, and the module the launcher builds for its CPU
# column is a .dll. The GPU column is still Metal's, so this is the plain C++
# launcher, the same one Linux builds.
CXX      := $(shell command -v clang++ >/dev/null 2>&1 && echo clang++ || echo c++)
CXXFLAGS := -std=c++20 -O2 -I runtime -I include
FW       :=
MAIN     := -x c++ src/main.mm
EXE      := .exe
else
# `clang++` if it is there, otherwise `c++`: gcc builds this too, it just wants
# `-fcoroutines`, which the launcher adds when its first compile asks for it.
CXX      := $(shell command -v clang++ >/dev/null 2>&1 && echo clang++ || echo c++)
CXXFLAGS := -std=c++20 -O2 -I runtime -I include
FW       := -ldl
MAIN     := -x c++ src/main.mm
endif
ROOT     := $(CURDIR)
BIN      := build/weft$(EXE)
GEN      := build/weft-gen$(EXE)

# Both tools, always: a translator change that is measured against a stale
# weft-gen reads as "the change did nothing".
#
# Each link goes to a temp name and is renamed into place, so a build that is
# interrupted mid-link cannot leave a `build/weft` whose timestamp says it is
# current while its contents are from the previous source. That failure reads as
# "the fix did nothing", which costs more than the rename does.
# The SPIR-V fixtures are assembled from the text beside them, so the built files
# are not in the repository. `spirv` is best effort, see the target, and so is
# safe here on a machine without the tools.
all: $(BIN) $(GEN) spirv

$(BIN): src/main.mm src/translate.cpp src/translate_ptx.cpp src/translate_spirv.cpp src/translate.hpp \
           src/translate_ptx.hpp src/translate_spirv.hpp src/weft_msl.hpp src/weft_warp_msl.hpp \
           src/weft_f64.hpp src/soft_math.hpp src/cuda_gpu.hpp runtime/weft_runtime.hpp \
           runtime/weft_runtime.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -DWEFT_ROOT='"$(ROOT)"' $(MAIN) src/translate.cpp src/translate_ptx.cpp \
	      src/translate_spirv.cpp $(FW) -o $(BIN).n && mv $(BIN).n $(BIN)

# The build-time translator: the same front ends, no Metal, no launcher. It turns a
# kernel file into a C++ translation unit you compile into your own program, so
# the CPU path costs nothing at run time.
$(GEN): src/gen.cpp src/translate.cpp src/translate_ptx.cpp src/translate_spirv.cpp \
                src/translate.hpp src/translate_ptx.hpp src/translate_spirv.hpp src/weft_msl.hpp \
                src/weft_warp_msl.hpp src/weft_f64.hpp src/soft_math.hpp
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -I include src/gen.cpp src/translate.cpp src/translate_ptx.cpp \
	      src/translate_spirv.cpp -o $(GEN).n && mv $(GEN).n $(GEN)

# The launcher's non-Metal column, built on a host that has Metal anyway. It is the
# code Linux and Windows compile, so this target is how that path is compiled and
# run where no such host is at hand.
build/weft-nometal: src/main.mm src/translate.cpp src/translate_ptx.cpp src/translate_spirv.cpp \
                    src/translate.hpp src/translate_ptx.hpp src/translate_spirv.hpp \
                    src/weft_msl.hpp src/weft_warp_msl.hpp src/weft_f64.hpp src/soft_math.hpp \
                    src/cuda_gpu.hpp runtime/weft_runtime.hpp runtime/weft_runtime.cpp
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -DWEFT_NO_METAL -x c++ -I runtime -I include \
	      -DWEFT_ROOT='"$(ROOT)"' src/main.mm src/translate.cpp src/translate_ptx.cpp \
	      src/translate_spirv.cpp -o build/weft-nometal.n && mv build/weft-nometal.n build/weft-nometal

nometal: build/weft-nometal

# The CUDA column against the stand-in driver in tools/fake_cuda.cpp: every call
# it makes is recorded and checked, so the binding is tested without CUDA.
cuda-calls: build/weft-nometal
	@sh tools/cuda_calls.sh

gen: $(GEN)

build/embed_kernel.cpp: $(GEN) kernels/basic.metal
	./$(GEN) kernels/basic.metal --kernel saxpy -o build/embed_kernel.cpp

# The linked path: the kernel is a translation unit in *this* binary, and the
# program calls it through include/weft/weft.h. No subprocess, no dylib, no device.
build/embed: examples/embed/embed.cpp build/embed_kernel.cpp runtime/weft_runtime.cpp
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -fno-strict-aliasing -fsigned-char -I include -I runtime examples/embed/embed.cpp \
	      build/embed_kernel.cpp runtime/weft_runtime.cpp -o build/embed

embed: build/embed
	@./build/embed build/embed

demo: $(BIN)
	@./$(BIN) run kernels/basic.metal --kernel saxpy      --grid 1048576 --buffers 1048576,1048576,1048576
	@./$(BIN) run kernels/basic.metal --kernel stencil5   --grid 1048576 --buffers 1048576,1048576
	@./$(BIN) run kernels/basic.metal --kernel reduce_sum --grid 1048576 --threads 256 --buffers 1048576,4096

LLC ?= $(shell command -v llc 2>/dev/null || echo /opt/homebrew/opt/llvm/bin/llc)

# Regenerate the PTX samples from LLVM IR with the NVPTX backend.
ptx:
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/saxpy.ll   -o kernels/cuda/saxpy.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/reverse.ll -o kernels/cuda/reverse.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/reduce.ll  -o kernels/cuda/reduce.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/scale.ll   -o kernels/cuda/scale.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/vecscale.ll -o kernels/cuda/vecscale.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/warpscan.ll  -o kernels/cuda/warpscan.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/blockreduc.ll -o kernels/cuda/block_reduce.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/rotate.ll  -o kernels/cuda/rotate.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/bsum.ll    -o kernels/cuda/bsum.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/hsum.ll    -o kernels/cuda/hsum.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/vote.ll    -o kernels/cuda/vote.ptx
	$(LLC) -march=nvptx64 -mcpu=sm_70 -O2 kernels/cuda/bits.ll    -o kernels/cuda/bits.ptx

# Assemble the hand-written SPIR-V modules and validate every one of them: the
# validator is the reference the front end is checked against.
SPIRV_AS  ?= $(shell command -v spirv-as 2>/dev/null || echo /opt/homebrew/bin/spirv-as)
SPIRV_VAL ?= $(shell command -v spirv-val 2>/dev/null || echo /opt/homebrew/bin/spirv-val)

spirv:
	# Both tools are external, so this is best effort: with them, every module is
	# rebuilt and validated; without, the files already on disk stand and the
	# message says which one was missing. `|| exit 1` inside the loops because a
	# for loop's status is its last command, so a module that failed to assemble
	# was being reported by nothing.
	# Each module is written under a temporary name and moved into place only after
	# it validates, so a module the tools reject cannot replace the good one that
	# was already there, a failed build leaves the tree as it found it and says
	# which file it stopped on, rather than half-writing the module it was on.
	@if ! command -v $(SPIRV_AS) >/dev/null 2>&1 && [ ! -x "$(SPIRV_AS)" ]; then \
	  echo "spirv: no spirv-as, so the modules already here are the ones used"; \
	elif ! command -v $(LLC) >/dev/null 2>&1 && [ ! -x "$(LLC)" ]; then \
	  echo "spirv: no llc, so the modules already here are the ones used"; \
	else \
	  val=""; \
	  if command -v $(SPIRV_VAL) >/dev/null 2>&1 || [ -x "$(SPIRV_VAL)" ]; then val=$(SPIRV_VAL); fi; \
	  tmp=kernels/spirv/.spirv.$$$$; ok=1; \
	  for f in kernels/spirv/*.spvasm; do \
	    $(SPIRV_AS) --target-env spv1.3 $$f -o $$tmp.bin || { \
	      echo "spirv: $$f did not assemble, so every module on disk stays as it was" >&2; ok=0; break; }; \
	    if [ -n "$$val" ] && ! $$val --target-env spv1.3 $$tmp.bin; then \
	      echo "spirv: $$f did not validate, so every module on disk stays as it was" >&2; ok=0; break; \
	    fi; \
	    mv -f $$tmp.bin $${f%.spvasm}.spv; \
	  done; \
	  if [ $$ok = 1 ]; then \
	    $(LLC) -march=spirv64 -filetype=obj -O2 kernels/cuda/spvloop.ll -o $$tmp.llvm || { \
	      echo "spirv: llc did not emit a module from kernels/cuda/spvloop.ll" >&2; ok=0; }; \
	    if [ $$ok = 1 ] && [ -n "$$val" ] && ! $$val --target-env spv1.4 $$tmp.llvm; then \
	      echo "spirv: llc's module did not validate, so kernels/spirv/llvm.spv keeps its last good copy" >&2; ok=0; \
	    fi; \
	    if [ $$ok = 1 ]; then mv -f $$tmp.llvm kernels/spirv/llvm.spv; fi; \
	  fi; \
	  rm -f $$tmp.bin $$tmp.llvm; \
	  if [ $$ok = 0 ]; then exit 1; fi; \
	fi

clean:
	rm -rf build

.PHONY: all demo ptx spirv gen embed nometal cuda-calls clean

# Regenerate the README Numbers table from a measurement rather than memory.
# BENCH=5 for a quick pass; the default is the best of 30.
numbers:
	@./tools/numbers.sh
