; Warp-wide sum with shfl.down and no data-dependent branches at all.
target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)

define void @bsum(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v0 = load float, ptr addrspace(1) %pin, align 4
  %b0 = bitcast float %v0 to i32
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %loop ]
  %acc = phi i32 [ %b0, %entry ], [ %accb, %loop ]
  %d = phi i32 [ 16, %entry ], [ %dnext, %loop ]
  %peer = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %acc, i32 %d, i32 31)
  %fa = bitcast i32 %acc to float
  %fb = bitcast i32 %peer to float
  %sum = fadd float %fa, %fb
  %accb = bitcast float %sum to i32
  %dnext = lshr i32 %d, 1
  %inext = add i32 %i, 1
  %cont = icmp ult i32 %inext, 5
  br i1 %cont, label %loop, label %done

done:
  %r = bitcast i32 %accb to float
  %po = getelementptr float, ptr addrspace(1) %out, i64 %gi
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @bsum, !"kernel", i32 1}
