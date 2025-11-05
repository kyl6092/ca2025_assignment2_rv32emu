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
    addi    sp, sp, -48
    sw      s0, 0(sp)
    sw      s1, 4(sp)
    sw      x18, 8(sp)
    sw      x19, 12(sp)
    sw      x20, 16(sp)
    
    # BLANK 1: Fix position at x2+20
    sw      x0, 20(sp)
    # BLANK 2: Fix position at x2+24
    sw      x0, 24(sp)
    # BLANK 3: Fix position at x2+28
    sw      x0, 28(sp)

    addi    x28, x0, 49
    sw      x28, 32(sp)
    addi    x28, x0, 50
    sw      x28, 36(sp)
    addi    x28, x0, 51
    sw      x28, 40(sp)
    addi    x28, x0, 10
    sw      x28, 44(sp)

    addi    s0, x0, 1
game_loop:
    addi    t0, x0, 8
    beq     s0, t0, finish_game

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
    addi    s1, x0, 0

    # BLANK 11: Mask for testing LSB
    andi    t1, t0, 1

    # BLANK 12: Branch if disk 0 moves
    bne     t1, x0, disk_found

    # BLANK 13: Set disk 1
    addi    s1, x0, 1

    # BLANK 14: Test second bit with proper mask
    andi    t1, t0, 2
    bne     t1, x0, disk_found

    # BLANK 15: Last disk number
    addi    s1, x0, 2

disk_found:
    # BLANK 16: Check impossible pattern (multiple bits)
    andi    x30, t0, 5
    addi    x31, x0, 5
    beq     x30, x31, pattern_match
    jal     x0, continue_move
pattern_match:
continue_move:
    # BLANK 17: Word-align disk index (multiply by what?)
    slli    t0, s1, 2

    # BLANK 18: Base offset for disk array
    addi    t0, t0, 20
    add     t0, sp, t0
    lw      x18, 0(t0)

    bne     s1, x0, handle_large

    # BLANK 19: Small disk moves by how many positions?
    addi    x19, x18, 2

    # BLANK 20: Number of pegs
    addi    t1, x0, 3
    blt     x19, t1, display_move
    sub     x19, x19, t1
    jal     x0, display_move

handle_large:
    # BLANK 21: Load reference disk position
    lw      t1, 20(sp)

    # BLANK 22: Sum of all peg indices (0+1+2)
    addi    x19, x0, 3
    sub     x19, x19, x18
    sub     x19, x19, t1

display_move:  
    # handle peg name
    la      x20, data
    add     t0, x20, x18
    lbu     x28, 0(t0)
    add     t2, x20, x19
    lbu     x29, 0(t2)
    # Print "Move Disk "
    la      a1, str1
    li      a2, 10
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # handle & Print disk name
    slli    x28, s1, 2
    addi    x28, x28, 32
    add     a1, sp, x28
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
    add     a1, sp, 44
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall

    # BLANK 26: Calculate storage offset
    slli    t0, s1, 2
    addi    t0, t0, 20
    add     t0, sp, t0

    # BLANK 27: Update disk position
    sw      x19, 0(t0)

    # BLANK 28-29: Increment counter and loop
    addi    s0, s0, 1
    j       game_loop

finish_game:
    addi    sp, sp, 48
    ret
.size hanoi,.-hanoi


