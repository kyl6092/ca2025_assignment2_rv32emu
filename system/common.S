.text

# === my_mul ===
.globl my_mul
.type  my_mul,%function
my_mul:
# a0 out1 (in1)
# a1 in2
# a2 (out2)
    add    x29, x0, a0
    beq    a0, zero, my_mul_ret
    addi   a0,  x0, 0
    addi   t0,  x0, 0
    addi   x19, x0, 0
    addi   x18, x0, 0
    beq    a1, zero, my_mul_ret
my_mul_loop:
    slli   x19, x19, 1
    add    x19, x19, x18
    andi   x28, a1, 1
    beq    x28, zero, my_mul_loop_1
    add    x28, a0, x29
    sltu   x18, x28, a0
    add    t0, t0, x18
    add    t0, t0, x19
    add    a0, x0, x28
my_mul_loop_1:
    slli   x28, x29, 1
    sltu   x18, x28, x29
    add    x29, x0, x28
    srli   a1, a1, 1
    bne    a1, zero, my_mul_loop
my_mul_ret:
    sw     t0, 0(a2)
    ret
.size my_mul,.-my_mul


# === my_clz ===
.global my_clz
.type   my_clz,%function
my_clz:
# a0 out (in)
    addi   t0, x0, 0
    beq    a0, zero, clz_rt
clz_step1:
    srli   x28, a0, 16
    bne    x28, x0, clz_step2
    addi   t0, t0, 16
    slli   a0, a0, 16 
clz_step2:
    srli   x28, a0, 24
    bne    x28, x0, clz_step3
    addi   t0, t0, 8
    slli   a0, a0, 8
clz_step3:
    srli   x28, a0, 28
    bne    x28, x0, clz_step4
    addi   t0, t0, 4
    slli   a0, a0, 4
clz_step4:
    srli   x28, a0, 30
    bne    x28, x0, clz_step5
    addi   t0, t0, 2
    slli   a0, a0, 2
clz_step5:
    srli   x28, a0, 31
    bne    x28, x0, clz_rt
    addi   t0, t0, 1
clz_rt:
    add    a0, x0, t0
    ret
.size my_clz,.-my_clz