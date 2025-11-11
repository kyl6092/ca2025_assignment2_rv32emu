.data
data:       .byte   0x41, 0x42, 0x43
str1:       .asciz  "Move Disk "
str2:       .asciz  " from "
str3:       .asciz  " to "

.text

# ====================================Function==========================================
.globl  hanoi
.type   hanoi,%function
hanoi:
# a0: number of disks
    addi    sp, sp, -56
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      s2, 8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    
    # BLANK 1: Fix position at x2+20
    sw      x0, 20(sp)
    # BLANK 2: Fix position at x2+24
    sw      x0, 24(sp)
    # BLANK 3: Fix position at x2+28
    sw      x0, 28(sp)

    sw      x0, 32(sp)

    addi    x28, x0, 49
    sw      x28, 36(sp)
    addi    x28, x0, 50
    sw      x28, 40(sp)
    addi    x28, x0, 51
    sw      x28, 44(sp)
    addi    x28, x0, 52
    sw      x28, 48(sp)
    addi    x28, x0, 10
    sw      x28, 52(sp)

    addi    s0, x0, 1
    sll    s5, s0, a0
    andi    s8, a0, 1
game_loop:
    # addi    t0, x0, 8
    beq     s0, s5, finish_game

    # Gray code formula: gray(n) = n XOR (n >> k)
    # BLANK 5: What is k for Gray code?
    srli    t0, s0, 1

    # BLANK 6: Complete Gray(n) calculation
    xor     t1, s0, t0

    # BLANK 7-8: Calculate previous value and its shift
    addi    t2, s0, -1
    srli    x28, t2, 1

    # BLANK 9: Generate Gray(n-1)
    xor     t2, t2, x28

    # BLANK 10: Which bits changed?
    xor     t0, t1, t2

    # Initialize disk number
    addi    s1, x0, 20

    # BLANK 11: Mask for testing LSB
    andi    t1, t0, 1

    # BLANK 12: Branch if disk 0 moves
    bne     t1, x0, disk_found

    # BLANK 13: Set disk 1
    addi    s1, s1, 4

    # BLANK 14: Test second bit with proper mask
    andi    t1, t0, 2
    bne     t1, x0, disk_found

    # BLANK 15: Last disk number
    addi    s1, s1, 4
    andi    t1, t0, 4
    bne     t1, x0, disk_found

    addi    s1, s1, 4


disk_found:
#     # BLANK 16: Check impossible pattern (multiple bits)
#     andi    x30, t0, 5
#     addi    x31, x0, 5
#     beq     x30, x31, pattern_match
#     jal     x0, continue_move
# pattern_match:
continue_move:
    # BLANK 17: Word-align disk index (multiply by what?)
    # slli    t0, s1, 2

    # BLANK 18: Base offset for disk array
    # addi    t0, t0, 20
    add     t0, sp, s1
    lw      s2, 0(t0)

    addi    s6, s1, -20
    bne     s6, x0, handle_large

    bne     s8, zero, odd
even:
    addi    s3, s2, -2
    bge     s3, zero, display_move
    addi    s3, s3, 3
    jal     x0, display_move
odd:
    addi    s3, s2, 2
    addi    t1, x0, 3
    blt     s3, t1, display_move
    sub     s3, s3, t1
    jal     x0, display_move

handle_large:
    # BLANK 21: Load reference disk position
    # in this case s2 is the desired value
    lw      t1, 20(sp)


    # BLANK 22: Sum of all peg indices (0+1+2)
    addi    s3, x0, 3
    sub     s3, s3, s2
    sub     s3, s3, t1

display_move:  
    # handle peg name
    la      s4, data
    add     t0, s4, s2
    add     t2, s4, s3
    # Print "Move Disk "
    la      a1, str1
    li      a2, 10
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # handle & Print disk name
    addi    s6, s1, 16
    add     a1, sp, s6
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # Print " from "
    la      a1, str2
    li      a2, 6
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # Print peg name
    addi    a1, t0, 0
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # Print " to "
    la      a1, str3
    li      a2, 4
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # Print peg name
    addi    a1, t2, 0
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # Print newline
    add     a1, sp, 52
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall

    # BLANK 26: Calculate storage offset
    # slli    t0, s1, 2
    # addi    t0, t0, 20
    add     t0, sp, s1

    # BLANK 27: Update disk position
    sw      s3, 0(t0)

    # BLANK 28-29: Increment counter and loop
    addi    s0, s0, 1
    j       game_loop

finish_game:
    lw      s0, 0(x2)
    lw      s1, 4(x2)
    lw      s2, 8(x2)
    lw      s3, 12(x2)
    lw      s4, 16(x2)
    addi    sp, sp, 56
    ret
.size hanoi,.-hanoi


