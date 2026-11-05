; Warp-level inclusive scan, the standard CUDA idiom (Harris's scan), written
; as LLVM IR so the NVPTX backend emits the shuffles the way a compiler does.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32)

define void @warpscan(ptr addrspace(1) %in, ptr addrspace(1) %out) {
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
  %lane = and i32 %tid, 31
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %merge ]
  %acc = phi i32 [ %b0, %entry ], [ %accnext, %merge ]
  %sh = shl i32 1, %i
  %peer = call i32 @llvm.nvvm.shfl.sync.up.i32(i32 -1, i32 %acc, i32 %sh, i32 0)
  %ok = icmp uge i32 %lane, %sh
  br i1 %ok, label %add, label %merge

add:
  %fa = bitcast i32 %acc to float
  %fb = bitcast i32 %peer to float
  %sum = fadd float %fa, %fb
  %bsum = bitcast float %sum to i32
  br label %merge

merge:
  %accnext = phi i32 [ %bsum, %add ], [ %acc, %loop ]
  %inext = add i32 %i, 1
  %cont = icmp ult i32 %inext, 5
  br i1 %cont, label %loop, label %done

done:
  %r = bitcast i32 %accnext to float
  %po = getelementptr float, ptr addrspace(1) %out, i64 %gi
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @warpscan, !"kernel", i32 1}
