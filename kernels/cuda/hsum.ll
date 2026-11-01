; Warp sum with heavy arithmetic per stage (no data-dependent branches).
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
define void @hsum(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v0 = load float, ptr addrspace(1) %pin, align 4
  br label %loop
loop:
  %acc = phi float [ %v0, %entry ], [ %m11, %loop ]
  %d = phi i32 [ 16, %entry ], [ %dnext, %loop ]
  %bi = bitcast float %acc to i32
  %peer = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %bi, i32 %d, i32 31)
  %pf = bitcast i32 %peer to float
  %accnext = fadd float %acc, %pf
  %m0 = fmul float %accnext, 1.000100e+00
  %m1 = fmul float %m0, 1.000100e+00
  %m2 = fmul float %m1, 1.000100e+00
  %m3 = fmul float %m2, 1.000100e+00
  %m4 = fmul float %m3, 1.000100e+00
  %m5 = fmul float %m4, 1.000100e+00
  %m6 = fmul float %m5, 1.000100e+00
  %m7 = fmul float %m6, 1.000100e+00
  %m8 = fmul float %m7, 1.000100e+00
  %m9 = fmul float %m8, 1.000100e+00
  %m10 = fmul float %m9, 1.000100e+00
  %m11 = fmul float %m10, 1.000100e+00
  %dnext = lshr i32 %d, 1
  %cont = icmp sgt i32 %dnext, 0
  br i1 %cont, label %loop, label %done
done:
  %po = getelementptr float, ptr addrspace(1) %out, i64 %gi
  store float %m11, ptr addrspace(1) %po, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{ptr @hsum, !"kernel", i32 1}
