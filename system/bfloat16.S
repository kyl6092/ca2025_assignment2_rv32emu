.data

mask_value:
    .word  0xFFFFFFFF

.text

# ====================================Function==========================================
# === float32_to_bfloat16 ===
.globl f32_to_bf16
.type  f32_to_bf16,%function
f32_to_bf16:
# a0 out (in)
    lw     t1, mask_value
    add    x28, x0, a0                  # move argument f32 to x28
    srli   x29, a0, 16                  # make shared x29 for (f32>>16)
    srli   x28, a0, 23                  # exponent field at LSB
    andi   x28, x28, 0xFF               # mask exponent
    addi   x28, x28, -0xFF              # result==0 equivant to exponent==0xFF
    bne    x28, zero, exp_non_ff        #
    slli   x28, x29, 16                 # eliminate upper 16-bits
    srli   a0, x28, 16                  # a0 = (bits>>16) & 0xFFFF
    ret
exp_non_ff:
    andi   x28, x29, 1                  # (f32>>16)&1
    srli   x29, t1, 17                  # make constant 0x7FFF
    add    x28, x28, x29
    add    a0, x28, a0
    srli   a0, a0, 16                   # f32 >> 16
    ret
.size f32_to_bf16,.-f32_to_bf16

# # === bfloat16_to_float32 ===
.globl bf16_to_f32
.type  bf16_to_f32,%function
bf16_to_f32:
# a0 out (in)
    slli   a0, a0, 16
    ret
.size bf16_to_f32,.-bf16_to_f32

# # === bf16_is_inf or f32_is_inf ===
.globl is_inf
.type  is_inf,%function
is_inf:
# a0 out (in)
# a3 mantissa mask offset
# a4 exponent mask offset
    srl    x29, t1, a3                  # mantissa mask 0x000007F
    and    x29, a0, x29
    bne    x29, zero, inf_fail_branch
    srl    x29, a0, a4                  # make exponent field at LSB
    andi   x29, x29, 0xFF               # make value uneffected by sign bit
    addi   x29, x29, -0xFF              # result==0 equivant to exponent==0xFF
    bne    x29, zero, inf_fail_branch
    addi   a0, x0, 1                    # The value is infinity
    ret
inf_fail_branch:
    addi   a0, x0, 0                    # The value is not infinity
    ret
.size is_inf,.-is_inf

# === bf16_is_nan or f32_is_nan ===
.globl is_nan
.type  is_nan,%function
is_nan:
# a0 out (in)
# a3 mantissa mask offset
# a4 exponent mask offset
    srl    x29, t1, a3                  # mantissa mask 0x000007F
    and    x29, a0, x29
    beq    x29, zero, nan_fail_branch
    srl    x29, a0, a4                  # make exponent field at LSB
    andi   x29, x29, 0xFF               # make value uneffected by sign bit
    addi   x29, x29, -0xFF              # result==0 equivant to exponent==0xFF
    bne    x29, zero, nan_fail_branch
    addi   a0, x0, 1                    # The value is nan
    ret
nan_fail_branch:
    addi   a0, x0, 0                    # The value is a number !
    ret
.size is_nan,.-is_nan

# === bf16_is_zero or f32_is_zero ===
.globl is_zero
.type  is_zero,%function
is_zero:
# a0 out (in)
# a3 mantissa+exp mask offset
    srl    x29, t1, a3
    and    a0, a0, x29                  # exp+mant mask
    beq    a0, zero, zero_branch
    addi   a0, x0, 0                    # The value is non-zero
    ret
zero_branch:
    addi   a0, x0, 1                    # The value is zero
    ret
.size is_zero,.-is_zero

# === bf16_eq ===
.globl is_eq
.type  is_eq,%function
is_eq:
# a0 out (in1)
# a1 in2
# a3 mantissa mask offset
# a4 exponent mask offset
# a5 sign mask offset
    addi   sp, sp, -12                  # Increase stack space
    sw     ra, 8(sp)                    # Save return addr
    sw     a1, 4(sp)                    # Save argument (oper2)
    sw     a0, 0(sp)                    # Save argument (oper1)
    jal    ra, is_nan                   # FuncCall for is_nan
    bne    a0, zero, EQ_false
    add    a0, x0, a1                   # Load oper2
    jal    ra, is_nan                   # FuncCall for is_nan
    bne    a0, zero, EQ_false
    lw     a0, 0(sp)                    # Load oper1
    addi   x29, x0, 32
    sub    a3, x29, a5                  # Load argument for bf16_is_zero (17=32-(7+8))
    jal    ra, is_zero                  # FuncCall for bf16_is_zero
    beq    a0, zero, EQ_CAL             # a0 = 1: is_zero # a0=0: is_non_zero
    lw     a0, 4(sp)                    # Load oper2
    jal    ra, is_zero                  # FuncCall for bf16_is_zero
    beq    a0, zero, EQ_CAL
    addi   a0, x0, 1                    # Return True
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
EQ_CAL:
    lw     x28, 0(sp)                   # Load oper1
    lw     x29, 4(sp)                   # Load oper2
    bne    x28, x29, EQ_false
    addi   a0, x0, 1                    # Return True
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
EQ_false:
    addi   a0, x0, 0                    # Return False
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
.size is_eq,.-is_eq
# === bf16_lt ===
.globl is_lt
.type  is_lt,%function
is_lt:
# a0 out (in1)
# a1 in2
# a3 mantissa mask offset
# a4 exponent mask offset
# a5 sign mask offset
    addi   sp, sp, -12                  # Increase stack space
    sw     ra, 8(sp)                    # Save return addr
    sw     a1, 4(sp)                    # Save argument (oper2)
    sw     a0, 0(sp)                    # Save argument (oper1)
    jal    ra, is_nan                   # FuncCall for bf16_is_nan
    bne    a0, zero, LT_false
    add    a0, x0, a1                   # Load oper2
    jal    ra, is_nan                   # FuncCall for bf16_is_nan
    bne    a0, zero, LT_false
    lw     a0, 0(sp)                    # Load oper1
    addi   x29, x0, 32
    sub    a3, x29, a5                  # Load argument for bf16_is_zero (17=32-(7+8))
    jal    ra, is_zero                  # FuncCall for bf16_is_zero
    beq    a0, zero, LT_CAL             # a0 = 1: is_zero # a0=0: is_non_zero
    lw     a0, 4(sp)                    # Load oper2
    jal    ra, is_zero                  # FuncCall for bf16_is_zero
    beq    a0, zero, LT_CAL
    j      LT_false
LT_CAL:
    lw     x28, 0(sp)                   # Load oper1
    lw     x29, 4(sp)                   # Load oper2
    srl    x30, x28, a5                 # extract MSB
    andi   x30, x30, 1                  # extract MSB
    srl    x31, x29, a5                 # extract MSB
    andi   x31, x31, 1                  # extract MSB
    beq    x31, x30, LT_SIGN_SAME       # (sign_oper1 != sign_oper2)
    blt    x30, x31, LT_false
    addi   a0, x0, 1                    # Return True
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
LT_SIGN_SAME:
    beq    x30, zero, LT_CAL_UNSIGN     # (is sign_oper1 negative?)
    blt    x28, x29, LT_false
    addi   a0, x0, 1                    # Return True
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
LT_CAL_UNSIGN:
    blt    x29, x28, LT_false
    add    a0, x0, x28
    add    a1, x0, x29
    jal    ra, is_eq
    bne    a0, zero, LT_false
    addi   a0, x0, 1
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
LT_false:
    addi   a0, x0, 0                    # Return False
    lw     ra, 8(sp)                    # Restore return addr
    addi   sp, sp, 12                   # Decrease stack space
    ret
.size is_lt,.-is_lt

# === bf16_gt ===
    .globl is_gt
    .type  is_gt,%function
is_gt:
# a0 out (in1)
# a1 in2
# a3 mantissa mask offset
# a4 exponent mask offset
# a5 sign mask offset
    addi   sp, sp, -4                   # Increase stack space
    sw     ra, 0(sp)                    # Save return addr
    add    x28, x0, a0                  # tmp=a0
    add    a0, x0, a1                   # a0=a1
    add    a1, x0, x28                  # a1=tmp
    jal    ra, is_lt
    lw     ra, 0(sp)
    addi   sp, sp, 4
    ret
.size is_gt,.-is_gt

# === my_add ===
    .globl my_add
    .type  my_add,%function
my_add:
# a0 out (in1)
# a1 in2
# a3 mantissa offset
# a4 exponent offset
# a5 sign offset
# x28, x29 tmp 
# x30, x31 sign
# x26, x27 exp
# x24, x25 mant
# x23 result sign
# x22 result exp
# x21 result mant
    lw     t1, mask_value
    srli   x28, t1, 24                  # 0xFF
    srl    x30, a0, a5                  # extract sign bit
    andi   x30, x30, 1                  # extract sign bit masking
    srl    x31, a1, a5                  # extract sign bit
    andi   x31, x31, 1                  # extract sign bit masking
    srl    x26, a0, a4                  # extract exponent
    andi   x26, x26, 0xFF               # extract exponent masking
    srl    x27, a1, a4                  # extract exponent
    andi   x27, x27, 0xFF               # extract exponent masking
    srl    x29, t1, a3                  # mantissa mask
    and    x24, a0, x29                 # extract mantissa masking
    and    x25, a1, x29                 # extract mantissa masking
    bne    x26, x28, add_stage2         # exp_a == 0xFF
    bne    x24, zero, rt_a
    bne    x27, x28, rt_a
    bne    x25, zero, rt_b
    beq    x30, x31, rt_b
    j      rt_nan
add_stage2:
    beq    x27, x28, rt_b
    bne    x26, zero, add_stage3
    beq    x24, zero, rt_b
add_stage3:
    bne    x27, zero, add_stage4
    beq    x25, zero, rt_a
add_stage4:
    beq    x26, zero, add_stage5
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x24, x24, x29                # change manta
add_stage5:
    beq    x27, zero, add_cal
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x25, x25, x29                # change mantb
add_cal:
    sub    x29, x26, x27                # exp_diff
    blt    x29, zero, exp_diff_neg_zero
    add    x22, x0, x26                 # exp = expa
    addi   x21, x29, -8                 # x21 (tmp)
    blt    zero, x21, rt_a
    srl    x25, x25, x29
    j      add_stage6
exp_diff_neg_zero:
    beq    x29, zero, exp_diff_zero
    add    x22, x0, x27                 # exp = expb
    addi   x21, x29, 8
    blt    x21, zero, rt_b
    sub    x29, x0, x29
    srl    x24, x24, x29
    j      add_stage6
exp_diff_zero:
    add    x22, x0, x26
add_stage6:
    bne    x30, x31, add_sign_diff
    add    x23, x0, x30                 # sign = singa
    add    x21, x24, x25                # mant = manta + mantb
    addi   x29, x0, 1
    sll    x29, x29, a4
    slli   x29, x29, 1
    and    x29, x21, x29                # result mant adjust
    beq    x29, zero, rt_cal
    srli   x21, x21, 1
    addi   x22, x22, 1
    blt    x22, x28, rt_cal
    j      rt_inf
add_sign_diff:
    blt    x24, x25, add_bgta
    add    x23, x0, x30                 # sign = signa
    sub    x21, x24, x25                # mant = manta - mantb
    j      add_zero_judge
add_bgta:
    add    x23, x0, x31
    sub    x21, x25, x24                # mant = mantb - manta
add_zero_judge:
    bne    x21, zero, mant_chg
    add    a0, x0, x0
    ret
mant_chg:
    addi   x29, x0, 1
    sll    x29, x29, a4
    and    x29, x21, x29
    bne    x29, zero, rt_cal
    slli   x21, x21, 1
    addi   x22, x22, -1
    blt    zero, x21, mant_chg
    add    a0, x0, x0
    ret
rt_cal:
    slli   a0, x23, 8
    and    x22, x22, x28
    or     a0, a0, x22
    sll    a0, a0, a4
    srl    x29, t1, a3                  # mantissa mask
    and    x21, x21, x29
    or     a0, a0, x21
    ret

# === my_sub ===
.globl my_sub
.type  my_sub,%function
my_sub:
# a0 out (in1)
# a1 in2
# a3 mantissa offset
# a4 exponent offset
# a5 sign offset
    addi   sp, sp, -4
    sw     ra, 0(sp)
    addi   x28, x0, 1
    sll    x28, x28, a5                 # sign masking
    xor    a1, a1, x28
    jal    ra, my_add
    lw     ra, 0(sp)
    addi   sp, sp, 4
    ret

rt_a:
    add    a0, x0, a0
    ret
rt_b:
    add    a0, x0, a1
    ret


# === my_fp_mul ===
.globl my_fp_mul
.type  my_fp_mul,%function
my_fp_mul:
# a0 out (in1)
# a1 in2
# a3 mantissa mask offset
# a4 exponent mask offset
# a5 sign mask offset
# a6 operation offset
# x28, x29 tmp
# x30, x31 sign
# x26, x27 exp
# x24, x25 mant
# x23 result sign
# x22 result exp
# x21 result mant
# x20 exp_adjust
    addi   sp, sp, -12
    sw     ra, 8(sp)
    sw     a1, 4(sp)
    sw     a0, 0(sp)
    lw     t1, mask_value
    srli   x28, t1, 24                  # 0xFF
    srl    x30, a0, a5                  # extract sign bit
    andi   x30, x30, 1                  # extract sign bit masking
    srl    x31, a1, a5                  # extract sign bit
    andi   x31, x31, 1                  # extract sign bit masking
    srl    x26, a0, a4                  # extract exponent
    andi   x26, x26, 0xFF               # extract exponent masking
    srl    x27, a1, a4                  # extract exponent
    andi   x27, x27, 0xFF               # extract exponent masking
    srl    x29, t1, a3                  # mantissa mask
    and    x24, a0, x29                 # extract mantissa masking
    and    x25, a1, x29                 # extract mantissa masking
    xor    x23, x30, x31                # result sign

    bne    x26, x28, mul_stage2         # exp_a == 0xFF
    beq    x24, zero, mul_manta_zero    # manta is zero
    j      rt_mul_a
mul_manta_zero:
    bne    x27, zero, rt_mul_inf
    beq    x25, zero, rt_mul_nan
    j      rt_mul_inf
mul_stage2:
    bne    x27, x28, mul_stage3
    beq    x25, zero, mul_mantb_zero
    j      rt_mul_b
mul_mantb_zero:
    bne    x26, zero, rt_mul_inf
    beq    x24, zero, rt_mul_nan
    j      rt_mul_inf
mul_stage3:
    bne    x26, zero, mul_judge_zero
    beq    x24, zero, rt_mul_zero
mul_judge_zero:
    bne    x27, zero, mul_stage4
    beq    x25, zero, rt_mul_zero
mul_stage4:
# adjust exp
    add    x20, x0, x0
    beq    x26, zero, expa_loop
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x24, x24, x29
    j      mul_stage5
expa_loop:
    addi   x29, x0, 1
    sll    x29, x29, a4
    and    x29, x24, x29
    bne    x29, zero, expa_loop_after
    slli   x24, x24, 1
    addi   x20, x20, -1
    j      expa_loop
expa_loop_after:
    addi   x26, x0, 1
mul_stage5:
    beq    x27, zero, expb_loop
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x25, x25, x29
    j      mul_stage6
expb_loop:
    addi   x29, x0, 1
    sll    x29, x29, a4
    and    x29, x25, x29
    bne    x29, zero, expb_loop_after
    slli   x25, x25, 1
    addi   x20, x20, -1
    j      expb_loop
expb_loop_after:
    addi   x27, x0, 1
mul_stage6:
    addi   x29, x0, 48
    beq    a6, x29, f32_mul_handle
    add    a0, x0, x24
    add    a1, x0, x25
    jal    ra, my_mul
    add    x21, x0, a0
# mul x21, x24, x25
    j      bf16_mul_handle
f32_mul_handle:
    srli   a0, x24, 7
    srli   a1, x25, 7
    jal    ra, my_mul
    add    x21, x0, a0
# mul x21, x24, x25
bf16_mul_handle:
    add    x22, x26, x27
    add    x22, x22, x20
    addi   x22, x22, -127

    srli   x29, x21, 15                 # resolution of mantissa
    bne    x29, zero, mul_sign_neg
    srl    x21, x21, a4
    # srl    x29, t1, a3
    andi    x21, x21, 0x7F
    j      mul_stage7
mul_sign_neg:
    srl    x21, x21, a4
    srli   x21, x21, 1
    # srl    x29, t1, a3
    andi    x21, x21, 0x7F
    addi   x22, x22, 1
mul_stage7:
    addi   x29, x22, -0xFF
    bge    x29, zero, rt_mul_inf
    blt    zero, x22, mul_rt_cal
    addi   x29, x22, 6
    blt    x29, zero, rt_mul_zero
    sub    x29, zero, x22
    addi   x29, x29, 1
    srl    x21, x21, x29
    add    x22, x0, x0
mul_rt_cal:
    slli   a0, x23, 8
    andi   x29, x22, 0xFF
    or     a0, a0, x29
    sll    a0, a0, a4
    # srl    x29, t1, a3
    andi    x29, x21, 0x7F
    or     a0, a0, x29
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_mul_a:
    lw     a0, 0(sp)
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_mul_b:
    lw     a0, 4(sp)
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_mul_inf:
    slli   a0, x23, 8
    addi   a0, a0, 0xFF
    sll    a0, a0, a4
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_inf:
    slli   a0, x23, 8
    addi   a0, a0, 0xFF
    sll    a0, a0, a4
    ret
rt_mul_zero:
    sll    a0, x23, a5
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_zero:
    sll    a0, x23, a5
    ret
rt_mul_nan:
    addi   x28, x0, 0xFF
    slli   a0, x28, 1
    ori    a0, a0, 1
    addi   x29, a4, -1
    sll    a0, a0, x29
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
rt_nan:
    slli   a0, x28, 1
    ori    a0, a0, 1
    addi   x29, a4, -1
    sll    a0, a0, x29
    ret
.size my_fp_mul,.-my_fp_mul

# === my_div ===
    .globl my_div
    .type  my_div,%function
my_div:
# a0 out (in1)
# a1 in2
# a3 mantissa mask offset
# a4 exponent mask offset
# a5 sign mask offset
# a6 operation offset
# x28, x29 tmp
# x30, x31 sign
# x26, x27 exp
# x24, x25 mant
# x23 result sign
# x22 result exp
# x21 result mant
# x20 iteration
    lw     t1, mask_value
    srli   x28, t1, 24                  # 0xFF
    srl    x30, a0, a5                  # extract sign bit
    andi   x30, x30, 1                  # extract sign bit masking
    srl    x31, a1, a5                  # extract sign bit
    andi   x31, x31, 1                  # extract sign bit masking
    srl    x26, a0, a4                  # extract exponent
    andi   x26, x26, 0xFF               # extract exponent masking
    srl    x27, a1, a4                  # extract exponent
    andi   x27, x27, 0xFF               # extract exponent masking
    srl    x29, t1, a3                  # mantissa mask
    and    x24, a0, x29                 # extract mantissa masking
    and    x25, a1, x29                 # extract mantissa masking
    xor    x23, x30, x31                # result sign

    bne    x27, x28, div_stage2         # exp_b == 0xFF
    bne    x25, zero, rt_b
    bne    x26, x28, rt_zero
    beq    x24, zero, rt_nan
    j      rt_zero
div_stage2:
    bne    x27, zero, div_stage3
    bne    x25, zero, div_stage3
    bne    x26, zero, rt_inf
    beq    x24, zero, rt_nan
    j      rt_inf
div_stage3:
    bne    x26, x28, div_stage4
    bne    x24, zero, rt_a
    j      rt_inf
div_stage4:
    bne    x26, zero, div_stage5
    beq    x24, zero, rt_zero
div_stage5:
    beq    x26, zero, div_chg_mantb
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x24, x24, x29
div_chg_mantb:
    beq    x27, zero, div_stage6
    addi   x29, x0, 1
    sll    x29, x29, a4
    or     x25, x25, x29
div_stage6:
    addi   x29, x0, 48
    beq    a6, x29, f32_div_handle
    slli   x24, x24, 15                 # dividend
    j      bf16_div_handle
f32_div_handle:
    srli   x24, x24, 7
    slli   x24, x24, 15
bf16_div_handle:
    add    x21, x0, x0                  # quotient
    addi   x20, x0, 16
quo_cal:
    addi   x20, x20, -1
    blt    x20, zero, div_stage7
    slli   x21, x21, 1
    sll    x29, x25, x20
    blt    x24, x29, quo_cal
    sub    x24, x24, x29
    ori    x21, x21, 1
    j      quo_cal
div_stage7:
    sub    x22, x26, x27                # result exp
    addi   x22, x22, 127
    bne    x26, zero, div_chg_exp
    addi   x22, x22, -1
div_chg_exp:
    bne    x27, zero, div_chg_quo1
    addi   x22, x22, 1
div_chg_quo1:
    srli   x29, x21, 15
    beq    x29, zero, div_chg_quo2
    srli   x21, x21, 8
    j      rt_div_cal
div_chg_quo2:
    srli   x29, x21, 15
    bne    x29, zero, div_chg_quo_after
    addi   x29, x22, -1
    bge    zero, x29, div_chg_quo_after
    slli   x21, x21, 1
    addi   x22, x22, -1
    j      div_chg_quo2
div_chg_quo_after:
    srli   x21, x21, 8
rt_div_cal:
    addi   x29, x22, -0xFF
    bge    x29, zero, rt_inf
    bge    zero, x22, rt_zero
    slli   a0, x23, 8
    andi   x29, x22, 0xFF
    or     a0, a0, x29
    sll    a0, a0, a4
    srl    x29, t1, a3
    and    x29, x21, x29
    or     a0, a0, x29
    ret
.size my_div,.-my_div

# === my_sqrt ===
.globl my_sqrt
.type  my_sqrt,%function
my_sqrt:
# a0 out (in)
# x28, x29 tmp
# x30 sign
# x26 exp
# x24 mant
# x25 x27 x31 low high result
# x20 sq
# x22 result exp
# x21 result mant
    addi   sp, sp, -12
    sw     ra, 8(sp)
    sw     a0, 0(sp)
    lw     t1, mask_value
    srli   x28, t1, 24                  # 0xFF
    srli   x30, a0, 15                  # sign
    andi   x30, x30, 1
    srli   x26, a0, 7                   # exp
    andi    x26, x26, 0xFF
    srli   x29, t1, 25                  # 0x7F
    and    x24, a0, x29
    add    x23, x0, x0
sqrt_case1:
    bne    x26, x28, sqrt_case2
    bne    x24, zero, rt_mul_a
    bne    x30, zero, rt_mul_nan
    j      rt_mul_a
sqrt_case2:
    bne    x26, zero, sqrt_case3
    beq    x24, zero, rt_mul_zero
sqrt_case3:
    bne    x30, zero, rt_mul_nan
sqrt_case4:
    beq    x26, zero, rt_mul_zero
    addi   x22, x26, -127               # e
    ori    x24, x24, 0x80               # m
    andi   x29, x22, 1
    bne    x29, zero, sqrt_exp_odd
sqrt_exp_even:
    srai   x22, x22, 1
    addi   x22, x22, 127
    j      binary_search
sqrt_exp_odd:
    slli   x24, x24, 1
    addi   x22, x22, -1
    srai   x22, x22, 1
    addi   x22, x22, 127
binary_search:
    addi   x25, x0, 90                  # low
    addi   x27, x0, 256                 # high
    addi   x31, x0, 128                 # result
binary_search_loop:
    blt    x27, x25, sqrt_normalize
    add    x29, x25, x27
    srai   x31, x29, 1
    add    a0, x0, x31
    add    a1, x0, x31
    jal    ra, my_mul
    srai   x20, a0, 7
# mul x20, x29, x29
    blt    x24, x20, sqrt_lower         # m < sq
# add    x31, x0, x29                 # result=mid
    addi   x25, x31, 1
    j      binary_search_loop
sqrt_lower:
    addi   x27, x31, -1
    j      binary_search_loop

sqrt_normalize:
    addi   x29, x31, -256
    blt    x29, zero, sqrt_cal
    srli   x31, x31, 1
    addi   x22, x22, 1
sqrt_cal:
    addi   x29, x22, -0xFF
    bge    x29, zero, sqrt_cal_case
    bge    zero, x22, rt_mul_zero
    andi   a0, x22, 0xFF
    slli   a0, a0, 7
    andi   x29, x31, 0x7F
    or     a0, a0, x29
    lw     ra, 8(sp)
    addi   sp, sp, 12
    ret
sqrt_cal_case:
    bne    x29, zero, rt_mul_inf
    andi   x29, x31, 0x7F               # new mant
    bne    x29, zero, rt_mul_nan
    j      rt_mul_inf
.size my_sqrt,.-my_sqrt