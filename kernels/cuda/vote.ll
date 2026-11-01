; Warp votes. Every lane's answer is fixed by its lane id, so the expected output
; is a constant: ballot of the even lanes is 0x55555555, all(lane < 32) is 1,
; any(lane == 7) is 1, and uni(lane < 32) is 1.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
declare i1 @llvm.nvvm.vote.all.sync(i32, i1)
declare i1 @llvm.nvvm.vote.any.sync(i32, i1)
declare i1 @llvm.nvvm.vote.uni.sync(i32, i1)

define void @vote(ptr addrspace(1) %out, ptr addrspace(1) %out2) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %lane = and i32 %tid, 31
  %l1 = and i32 %lane, 1
  %even = icmp eq i32 %l1, 0
  %bal = call i32 @llvm.nvvm.vote.ballot.sync(i32 -1, i1 %even)
  %po = getelementptr i32, ptr addrspace(1) %out, i64 %gi
  store i32 %bal, ptr addrspace(1) %po, align 4

  %lt = icmp slt i32 %lane, 32
  %al = call i1 @llvm.nvvm.vote.all.sync(i32 -1, i1 %lt)
  %u = call i1 @llvm.nvvm.vote.uni.sync(i32 -1, i1 %lt)
  %is7 = icmp eq i32 %lane, 7
  %ay = call i1 @llvm.nvvm.vote.any.sync(i32 -1, i1 %is7)

  %av = zext i1 %al to i32
  %uv = zext i1 %u to i32
  %yv = zext i1 %ay to i32
  %t = add i32 %av, %uv
  %r = add i32 %t, %yv
  %po2 = getelementptr i32, ptr addrspace(1) %out2, i64 %gi
  store i32 %r, ptr addrspace(1) %po2, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @vote, !"kernel", i32 1}
