; The one SPIR-V sample that LLVM's backend emits, rather than being hand-written:
; a serial sum with a phi loop, a scalar kernel parameter, an access chain, a
; load and an OpenCL ext-inst. LLVM's SPIR-V backend cannot be used for anything
; that reads a builtin id (it emits an invalid vectorless OpUConvert for
; get_global_id), so the identity of the thread comes from the loop, not the
; launch geometry.
target triple = "spirv64-unknown-unknown"

declare float @llvm.sqrt.f32(float)

define spir_kernel void @serial_sum(ptr addrspace(1) %a, ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %pi = getelementptr float, ptr addrspace(1) %a, i32 %i
  %v = load float, ptr addrspace(1) %pi
  %s = call float @llvm.sqrt.f32(float %v)
  %sum = fadd float %acc, %s
  %next = add i32 %i, 1
  %c = icmp ult i32 %next, %n
  br i1 %c, label %loop, label %done
done:
  store float %sum, ptr addrspace(1) %out
  ret void
}
