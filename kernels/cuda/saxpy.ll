; Source for the PTX test: out[gid] = 2.0f * a[gid] + 0.5f * b[gid]
; llc -march=nvptx64 -mcpu=sm_70 emits the .ptx from this.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @saxpy(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %i64 = sext i32 %gid to i64
  %pa = getelementptr float, ptr addrspace(1) %a, i64 %i64
  %va = load float, ptr addrspace(1) %pa, align 4
  %pb = getelementptr float, ptr addrspace(1) %b, i64 %i64
  %vb = load float, ptr addrspace(1) %pb, align 4
  %m1 = fmul float %va, 2.0
  %m2 = fmul float %vb, 0.5
  %r = fadd float %m1, %m2
  %po = getelementptr float, ptr addrspace(1) %out, i64 %i64
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @saxpy, !"kernel", i32 1}
