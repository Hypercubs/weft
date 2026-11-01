; out[gid] = in[gid] * k, guarded by a scalar parameter n.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @scale(ptr addrspace(1) %in, ptr addrspace(1) %out, float %k, i32 %n) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %ok = icmp slt i32 %gid, %n
  br i1 %ok, label %do, label %exit
do:
  %gi = sext i32 %gid to i64
  %p = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v = load float, ptr addrspace(1) %p, align 4
  %m = fmul float %v, %k
  %q = getelementptr float, ptr addrspace(1) %out, i64 %gi
  store float %m, ptr addrspace(1) %q, align 4
  br label %exit
exit:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @scale, !"kernel", i32 1}
