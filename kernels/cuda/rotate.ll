; out[gid] = in[gid rotated one lane up within its warp] — a shfl.idx.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)

define void @rotate(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %p = getelementptr i32, ptr addrspace(1) %in, i64 %gi
  %v = load i32, ptr addrspace(1) %p, align 4
  %lane = and i32 %tid, 31
  %next = add i32 %lane, 1
  %src = and i32 %next, 31
  %sh = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %v, i32 %src, i32 31)
  %q = getelementptr i32, ptr addrspace(1) %out, i64 %gi
  store i32 %sh, ptr addrspace(1) %q, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @rotate, !"kernel", i32 1}
