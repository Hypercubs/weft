; out[gid] = tile[tgsize - 1 - lid] after a threadgroup barrier.
target triple = "nvptx64-nvidia-cuda"

@tile = internal addrspace(3) global [256 x float] undef, align 4

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0()

define void @reverse(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v = load float, ptr addrspace(1) %pin, align 4
  %ps = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 %tid
  store float %v, ptr addrspace(3) %ps, align 4
  call void @llvm.nvvm.barrier0()
  %last = sub i32 %ntid, 1
  %ridx = sub i32 %last, %tid
  %pr = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 %ridx
  %w = load float, ptr addrspace(3) %pr, align 4
  %pout = getelementptr float, ptr addrspace(1) %out, i64 %gi
  store float %w, ptr addrspace(1) %pout, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @reverse, !"kernel", i32 1}
