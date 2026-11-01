; Tree reduction in threadgroup memory, one barrier per level — the same
; algorithm as reduce_sum in basic.metal, but emitted by LLVM.
target triple = "nvptx64-nvidia-cuda"

@tile = internal addrspace(3) global [256 x float] undef, align 4

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0()

define void @reduce(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v = load float, ptr addrspace(1) %pin, align 4
  %pt0 = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 %tid
  store float %v, ptr addrspace(3) %pt0, align 4
  call void @llvm.nvvm.barrier0()
  br label %loop

loop:
  %stride = phi i32 [ 128, %entry ], [ %snext, %latch ]
  %cmp = icmp slt i32 %tid, %stride
  br i1 %cmp, label %doit, label %after

doit:
  %pt1 = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 %tid
  %a = load float, ptr addrspace(3) %pt1, align 4
  %idx2 = add i32 %tid, %stride
  %pt2 = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 %idx2
  %b = load float, ptr addrspace(3) %pt2, align 4
  %sum = fadd float %a, %b
  store float %sum, ptr addrspace(3) %pt1, align 4
  br label %after

after:
  call void @llvm.nvvm.barrier0()
  br label %latch

latch:
  %snext = lshr i32 %stride, 1
  %cont = icmp sgt i32 %snext, 0
  br i1 %cont, label %loop, label %done

done:
  %is0 = icmp eq i32 %tid, 0
  br i1 %is0, label %write, label %exit

write:
  %pt3 = getelementptr [256 x float], ptr addrspace(3) @tile, i32 0, i32 0
  %r = load float, ptr addrspace(3) %pt3, align 4
  %co = sext i32 %ctaid to i64
  %po = getelementptr float, ptr addrspace(1) %out, i64 %co
  store float %r, ptr addrspace(1) %po, align 4
  br label %exit

exit:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @reduce, !"kernel", i32 1}
