; A whole block reduction, the way CUB does it: shfl_down within each warp, one
; shared-memory slot per warp, a barrier, then warp 0 reduces the warp sums.
; Same algorithm as block_reduce in basic.metal, emitted by LLVM so the shuffles
; and the barrier are real compiler output rather than hand-written PTX.
target triple = "nvptx64-nvidia-cuda"

@warp_sums = internal addrspace(3) global [32 x float] undef, align 4

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare void @llvm.nvvm.barrier0()
declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)

define void @block_reduce(ptr addrspace(1) %in, ptr addrspace(1) %out) {
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
  br label %wloop

; ---- warp reduction ----
wloop:
  %wi = phi i32 [ 0, %entry ], [ %winext, %wmerge ]
  %wacc = phi i32 [ %b0, %entry ], [ %wnext, %wmerge ]
  %wd = phi i32 [ 16, %entry ], [ %wdnext, %wmerge ]
  %wpeer = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %wacc, i32 %wd, i32 31)
  %wfa = bitcast i32 %wacc to float
  %wfb = bitcast i32 %wpeer to float
  %wsum = fadd float %wfa, %wfb
  %waccb = bitcast float %wsum to i32
  br label %wmerge

wmerge:
  %wnext = phi i32 [ %waccb, %wloop ]
  %wdnext = lshr i32 %wd, 1
  %winext = add i32 %wi, 1
  %wcont = icmp ult i32 %winext, 5
  br i1 %wcont, label %wloop, label %publish

; ---- one slot per warp, then a barrier ----
publish:
  %lane = and i32 %tid, 31
  %is0 = icmp eq i32 %lane, 0
  %wid = lshr i32 %tid, 5
  br i1 %is0, label %store, label %synced

store:
  %pw = getelementptr [32 x float], ptr addrspace(3) @warp_sums, i32 0, i32 %wid
  store float %wsum, ptr addrspace(3) %pw, align 4
  br label %synced

synced:
  call void @llvm.nvvm.barrier0()
  %nwarps = add i32 %ntid, 31
  %nwarps2 = lshr i32 %nwarps, 5
  %inwarp = icmp ult i32 %tid, 32
  br i1 %inwarp, label %load, label %exit

; ---- warp 0 reduces the warp sums ----
load:
  %hassum = icmp ult i32 %tid, %nwarps2
  %pr = getelementptr [32 x float], ptr addrspace(3) @warp_sums, i32 0, i32 %tid
  %rs = load float, ptr addrspace(3) %pr, align 4
  %t0 = select i1 %hassum, float %rs, float 0.000000e+00
  %tb0 = bitcast float %t0 to i32
  br label %tloop

tloop:
  %ti = phi i32 [ 0, %load ], [ %tinext, %tmerge ]
  %tacc = phi i32 [ %tb0, %load ], [ %tnext, %tmerge ]
  %td = phi i32 [ 16, %load ], [ %tdnext, %tmerge ]
  %tpeer = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 -1, i32 %tacc, i32 %td, i32 31)
  %tfa = bitcast i32 %tacc to float
  %tfb = bitcast i32 %tpeer to float
  %tsum = fadd float %tfa, %tfb
  %taccb = bitcast float %tsum to i32
  br label %tmerge

tmerge:
  %tnext = phi i32 [ %taccb, %tloop ]
  %tdnext = lshr i32 %td, 1
  %tinext = add i32 %ti, 1
  %tcont = icmp ult i32 %tinext, 5
  br i1 %tcont, label %tloop, label %writer

writer:
  %last = icmp eq i32 %tid, 0
  br i1 %last, label %emit, label %exit

emit:
  %r = bitcast i32 %tnext to float
  %co = sext i32 %ctaid to i64
  %po = getelementptr float, ptr addrspace(1) %out, i64 %co
  store float %r, ptr addrspace(1) %po, align 4
  br label %exit

exit:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @block_reduce, !"kernel", i32 1}
