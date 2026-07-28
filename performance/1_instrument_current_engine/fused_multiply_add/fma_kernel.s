# =====================================================================
#  fma_kernel.s  —  the tightest fp64 FMA loop the Skylake core allows.
#  Computes nothing useful; its ONLY job is to keep ports 0 & 1 saturated
#  so the measured GFLOP/s approaches the hardware ceiling from tab 02.
#
#  Instruction ref (what vfmadd231pd does, lane by lane):
#    https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd
#  Why 0.5 recip-throughput = 2/cycle (the "2 FMA ports" fact):
#    https://www.agner.org/optimize/instruction_tables.pdf   (search VFMADD)
#  AT&T vs Intel assembly syntax (operand order is REVERSED here):
#    https://en.wikipedia.org/wiki/X86_assembly_language#Syntax
# =====================================================================

    .text
    .globl _fma_kernel            # macOS prefixes C symbols with '_'. On Linux
                                  # use 'fma_kernel' (no underscore) instead.

# double fma_kernel(uint64_t iters);
#   arg  iters -> %rdi   (System V AMD64 calling convention: 1st int arg)
#   ret  double -> %xmm0
#   FLOPs performed = iters * 10 accumulators * 4 lanes * 2 (mul+add)
_fma_kernel:
    vbroadcastsd Lc(%rip), %ymm10   # ymm10 = [c,c,c,c], a tiny fp64 constant in all 4 lanes
    vmovapd      %ymm10, %ymm11     # ymm11 = same; we multiply ymm10*ymm11 each step

    # 10 independent accumulators, all zeroed. xorpd of a reg with itself = 0.
    vxorpd %ymm0,%ymm0,%ymm0
    vxorpd %ymm1,%ymm1,%ymm1
    vxorpd %ymm2,%ymm2,%ymm2
    vxorpd %ymm3,%ymm3,%ymm3
    vxorpd %ymm4,%ymm4,%ymm4
    vxorpd %ymm5,%ymm5,%ymm5
    vxorpd %ymm6,%ymm6,%ymm6
    vxorpd %ymm7,%ymm7,%ymm7
    vxorpd %ymm8,%ymm8,%ymm8
    vxorpd %ymm9,%ymm9,%ymm9

    testq %rdi, %rdi            # if iters == 0, skip the loop
    je    Ldone
Lloop:
    # AT&T operand order: vfmadd231pd  SRC2, SRC1, DST
    #   computes  DST = SRC1*SRC2 + DST   -> here  ymmN = ymm10*ymm11 + ymmN
    # 10 of them = 5 cycles of work on 2 ports; the next iteration's copies
    # can start before these finish, so both FMA ports stay 100% busy.
    vfmadd231pd %ymm11,%ymm10,%ymm0
    vfmadd231pd %ymm11,%ymm10,%ymm1
    vfmadd231pd %ymm11,%ymm10,%ymm2
    vfmadd231pd %ymm11,%ymm10,%ymm3
    vfmadd231pd %ymm11,%ymm10,%ymm4
    vfmadd231pd %ymm11,%ymm10,%ymm5
    vfmadd231pd %ymm11,%ymm10,%ymm6
    vfmadd231pd %ymm11,%ymm10,%ymm7
    vfmadd231pd %ymm11,%ymm10,%ymm8
    vfmadd231pd %ymm11,%ymm10,%ymm9
    decq %rdi                    # iters--
    jne  Lloop                    # loop while iters != 0
Ldone:
    # Reduce the 10 vector accumulators down to one scalar so the compiler
    # can't delete the whole loop as "dead code". Correctness of the sum
    # doesn't matter — only that the result is USED (returned).
    vaddpd %ymm1,%ymm0,%ymm0
    vaddpd %ymm3,%ymm2,%ymm2
    vaddpd %ymm5,%ymm4,%ymm4
    vaddpd %ymm7,%ymm6,%ymm6
    vaddpd %ymm9,%ymm8,%ymm8
    vaddpd %ymm2,%ymm0,%ymm0
    vaddpd %ymm6,%ymm4,%ymm4
    vaddpd %ymm8,%ymm0,%ymm0
    vaddpd %ymm4,%ymm0,%ymm0   # now ymm0 holds 4 partial sums
    vextractf128 $1, %ymm0, %xmm1  # grab the high 128 bits (lanes 2,3)
    vaddpd %xmm1,%xmm0,%xmm0       # add to low 128 (lanes 0,1) -> 2 doubles
    vhaddpd %xmm0,%xmm0,%xmm0      # horizontal add -> 1 double, now in xmm0 = return
    vzeroupper                     # good hygiene: avoid AVX/SSE transition stalls
    ret

    .section __TEXT,__const       # (Linux: use  .section .rodata  instead)
    .p2align 3
Lc: .double 1.0000001            # tiny value: accumulators grow slowly, never overflow to inf
