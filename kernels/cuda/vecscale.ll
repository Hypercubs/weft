; Four elements per thread through a <4 x float> load/store, which the backend
; turns into ld.global.v4.f32 / st.global.v4.f32.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @vecscale(ptr addrspace(1) %in, ptr addrspace(1) %out, float %k) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = sext i32 %gid to i64
  %p = getelementptr <4 x float>, ptr addrspace(1) %in, i64 %gi
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %k4 = insertelement <4 x float> poison, float %k, i32 0
  %kb = shufflevector <4 x float> %k4, <4 x float> poison, <4 x i32> zeroinitializer
  %m = fmul <4 x float> %v, %kb
  %q = getelementptr <4 x float>, ptr addrspace(1) %out, i64 %gi
  store <4 x float> %m, ptr addrspace(1) %q, align 16
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @vecscale, !"kernel", i32 1}
