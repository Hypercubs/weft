; Integer bit manipulation the way LLVM's NVPTX backend really emits it: a
; rotate is a funnel shift, a division by a constant is a high-word multiply,
; a float-to-int is a rounding convert, and a byte permute is `prmt`. The Metal
; kernel `bits` computes the same values by other means, and check.sh compares
; the two byte for byte.
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
declare i32 @llvm.nvvm.prmt(i32, i32, i32)
declare float @llvm.rint.f32(float)
declare float @llvm.floor.f32(float)

define void @bits(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %gid = add i32 %base, %tid
  %gi = zext i32 %gid to i64
  %pin = getelementptr float, ptr addrspace(1) %in, i64 %gi
  %v = load float, ptr addrspace(1) %pin, align 4

  ; A value with a fraction and a sign: rounding modes must disagree on it.
  %m = fmul float %v, 1.000000e+03
  %f = fadd float %m, 1.000000e+02
  %i = fptosi float %f to i32

  ; A hash, and the two ways to move bits around inside it.
  %a = mul i32 %i, 2654435761
  %s = and i32 %i, 31
  %rot = call i32 @llvm.fshl.i32(i32 %a, i32 305419896, i32 %s)
  %rot2 = call i32 @llvm.fshr.i32(i32 305419896, i32 %a, i32 %s)

  ; Division by a constant: 32- and 64-bit, unsigned and signed.
  %h32 = udiv i32 %a, 3
  %a64 = zext i32 %a to i64
  %i64 = zext i32 %i to i64
  %big = mul i64 %a64, 4096
  %big2 = add i64 %big, %i64
  %h64 = udiv i64 %big2, 1000
  %h64m = and i64 %h64, 2147483647
  %h64t = trunc i64 %h64m to i32
  %sd = sdiv i64 %big2, -1000
  %sdm = and i64 %sd, 65535
  %sdt = trunc i64 %sdm to i32

  ; Byte permute: four nibbles pick four of the eight bytes of (a:b).
  %perm = call i32 @llvm.nvvm.prmt(i32 %a, i32 305419896, i32 8256)

  ; NaN test on a value that is a NaN in half the lanes.
  %odd = and i32 %a, 1
  %isz = icmp eq i32 %odd, 0
  %nb = select i1 %isz, i32 1065353216, i32 2143289344
  %nv = bitcast i32 %nb to float
  %isn = fcmp uno float %nv, %nv
  %nn = zext i1 %isn to i32

  ; Rounding converts, which are `cvt.rni` / `cvt.rmi` followed by `cvt.rzi`.
  %rn = call float @llvm.rint.f32(float %f)
  %irn = fptosi float %rn to i32
  %rm = call float @llvm.floor.f32(float %f)
  %irm = fptosi float %rm to i32

  %t1 = add i32 %i, %rot
  %t2 = add i32 %t1, %rot2
  %t3 = add i32 %t2, %h32
  %t4 = add i32 %t3, %h64t
  %t5 = add i32 %t4, %sdt
  %t6 = add i32 %t5, %perm
  %t7 = add i32 %t6, %nn
  %t8 = add i32 %t7, %irn
  %t9 = add i32 %t8, %irm

  %po = getelementptr i32, ptr addrspace(1) %out, i64 %gi
  store i32 %t9, ptr addrspace(1) %po, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @bits, !"kernel", i32 1}
