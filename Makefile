# The launcher is one file on every platform. On macOS its GPU column needs
# Metal + Foundation and it is an Objective-C++ file; elsewhere there is no GPU
# backend, the Metal-only parts are compiled out, and it is plain C++ that links
# against nothing but libdl.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
# Apple's clang shim by full path, not whatever `clang++` a Homebrew LLVM put
# first on PATH: that one cannot link the ObjC message sends this uses against
# Apple's frameworks, and the failure reads as a missing SDK.
CXX      := $(shell test -x /usr/bin/clang++ && echo /usr/bin/clang++ || echo clang++)
CXXFLAGS := -std=c++20 -O2 -fobjc-arc -I runtime -I include
FW       := -framework Metal -framework Foundation
MAIN     := src/main.mm
else
# `clang++` if it is there, otherwise `c++`: gcc builds this too, it just wants
# `-fcoroutines`, which the launcher adds when its first compile asks for it.
CXX      := $(shell command -v clang++ >/dev/null 2>&1 && echo clang++ || echo c++)
CXXFLAGS := -std=c++20 -O2 -I runtime -I include
FW       := -ldl
MAIN     := -x c++ src/main.mm
endif
ROOT     := $(CURDIR)

# Both tools, always: a translator change that is measured against a stale
# weft-gen reads as "the change did nothing".
#
# Each link goes to a temp name and is renamed into place, so a build that is
# interrupted mid-link cannot leave a `build/weft` whose timestamp says it is
# current while its contents are from the previous source. That failure reads as
# "the fix did nothing", which costs more than the rename does.
# The SPIR-V fixtures are assembled from the text beside them, so the built files
# are not in the repository. `spirv` is best effort — see the target — and so is
# safe here on a machine without the tools.
all: build/weft build/weft-gen spirv

build/weft: src/main.mm src/translate.cpp src/translate_ptx.cpp src/translate_spirv.cpp src/translate.hpp \
           src/translate_ptx.hpp src/translate_spirv.hpp src/weft_msl.hpp src/weft_warp_msl.hpp \
           src/soft_math.hpp runtime/weft_runtime.hpp runtime/weft_runtime.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -DWEFT_ROOT='"$(ROOT)"' $(MAIN) src/translate.cpp src/translate_ptx.cpp \
	      src/translate_spirv.cpp $(FW) -o build/weft.n && mv build/weft.n build/weft

# The build-time translator: the same front ends, no Metal, no launcher. It turns a
# kernel file into a C++ translation unit you compile into your own program, so
# the CPU path costs nothing at run time.
build/weft-gen: src/gen.cpp src/translate.cpp src/translate_ptx.cpp src/translate_spirv.cpp \
                src/translate.hpp src/translate_ptx.hpp src/translate_spirv.hpp src/weft_msl.hpp \
                src/weft_warp_msl.hpp src/soft_math.hpp
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -I include src/gen.cpp src/translate.cpp src/translate_ptx.cpp \
	      src/translate_spirv.cpp -o build/weft-gen.n && mv build/weft-gen.n build/weft-gen

gen: build/weft-gen

build/embed_kernel.cpp: build/weft-gen kernels/basic.metal
	./build/weft-gen kernels/basic.metal --kernel saxpy -o build/embed_kernel.cpp

# The linked path: the kernel is a translation unit in *this* binary, and the
# program calls it through include/weft/weft.h. No subprocess, no dylib, no device.
build/embed: examples/embed/embed.cpp build/embed_kernel.cpp runtime/weft_runtime.cpp
	@mkdir -p build
	$(CXX) -std=c++20 -O2 -fsigned-char -I include -I runtime examples/embed/embed.cpp \
	      build/embed_kernel.cpp runtime/weft_runtime.cpp -o build/embed

embed: build/embed
	@./build/embed build/embed

demo: build/weft
	@./build/weft run kernels/basic.metal --kernel saxpy      --grid 1048576 --buffers 1048576,1048576,1048576
	@./build/weft run kernels/basic.metal --kernel stencil5   --grid 1048576 --buffers 1048576,1048576
	@./build/weft run kernels/basic.metal --kernel reduce_sum --grid 1048576 --threads 256 --buffers 1048576,4096

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
	@if ! command -v $(SPIRV_AS) >/dev/null 2>&1 && [ ! -x "$(SPIRV_AS)" ]; then \
	  echo "spirv: no spirv-as, so the modules already here are the ones used"; \
	elif ! command -v $(LLC) >/dev/null 2>&1 && [ ! -x "$(LLC)" ]; then \
	  echo "spirv: no llc, so the modules already here are the ones used"; \
	else \
	  for f in kernels/spirv/*.spvasm; do \
	    $(SPIRV_AS) --target-env spv1.3 $$f -o $${f%.spvasm}.spv || exit 1; \
	  done && \
	  $(LLC) -march=spirv64 -filetype=obj -O2 kernels/cuda/spvloop.ll -o kernels/spirv/llvm.spv && \
	  if command -v $(SPIRV_VAL) >/dev/null 2>&1 || [ -x "$(SPIRV_VAL)" ]; then \
	    for f in kernels/spirv/*.spv; do \
	      case $$f in *llvm.spv) e=spv1.4 ;; *) e=spv1.3 ;; esac; \
	      $(SPIRV_VAL) --target-env $$e $$f || exit 1; \
	    done; \
	  fi; \
	fi

clean:
	rm -rf build

.PHONY: all demo ptx spirv gen embed clean

# Regenerate the README Numbers table from a measurement rather than memory.
# BENCH=5 for a quick pass; the default is the best of 30.
numbers:
	@./tools/numbers.sh
