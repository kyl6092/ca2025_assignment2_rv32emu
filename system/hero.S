.data

.text

# === Main Function ===
.globl  hero
.type   hero,%function
# a0 a (return)
# a1 b
# a2 c
hero:
    # li    t1, 0xFFFFFFFF
    addi  sp, sp, -24
    sw    ra, 0(sp)
    sw    a0, 4(sp)
    sw    a1, 8(sp)
    sw    a2, 12(sp)

    addi  a3, x0, 25
    addi  a4, x0, 7
    addi  a5, x0, 15
    addi  a6, x0, 15
    jal   ra, my_fp_mul
    lw    a0, 4(sp)
    lw    a1, 8(sp)
    jal   ra, my_add                         # a+b
    lw    a1, 12(sp)
    jal   ra, my_add                         # a+b+c
    li    a1, 0x4000
    jal   ra, my_div
    sw    a0, 16(sp)                         # S=(a+b+c)/2

    lw    a1, 4(sp)
    jal   ra, my_sub                         # s-a
    lw    a1, 16(sp)
    
    jal   ra, my_fp_mul                         
    sw    a0, 20(sp)                         # s*(s-a)

    lw    a0, 16(sp)
    lw    a1, 8(sp)
    jal   ra, my_sub                         # s-b
    lw    a1, 20(sp)
    jal   ra, my_fp_mul
    sw    a0, 20(sp)                         # s*(s-a)*(s-b)      

    lw    a0, 16(sp)
    lw    a1, 12(sp)
    jal   ra, my_sub                         # s-c
    lw    a1, 20(sp)
    jal   ra, my_fp_mul                      # s*(s-a)*(s-b)*(s-c)

    jal   ra, my_sqrt                        # (s*(s-a)*(s-b)*(s-c))^0.5
    lw    ra, 0(sp)
    addi  sp, sp, 24
    ret
.size hero,.-hero
