.data
data_peg:       .byte   0x41, 0x42, 0x43
data_disk:      .word   0x31, 0x32, 0x33, 0x34, 0x0a

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
    sw      s5, 20(sp)
    sw      s6, 24(sp)
    sw      s7, 28(sp)
    sw      x0, 32(sp)
    sw      x0, 36(sp)
    sw      x0, 40(sp)
    sw      x0, 44(sp)
    addi    s0, x0, 1
    sll     s5, s0, a0
    andi    s8, a0, 1
game_loop:
    beq     s0, s5, finish_game
    srli    t0, s0, 1
    xor     t1, s0, t0
    addi    t2, s0, -1
    srli    x28, t2, 1
    xor     t2, t2, x28
    xor     t0, t1, t2
    addi    s1, x0, 32
    andi    t1, t0, 1
    bne     t1, x0, disk_found
    addi    s1, s1, 4
    andi    t1, t0, 2
    bne     t1, x0, disk_found
    addi    s1, s1, 4
    andi    t1, t0, 4
    bne     t1, x0, disk_found
    addi    s1, s1, 4
disk_found:
continue_move:
    add     t0, sp, s1
    lw      s2, 0(t0)
    addi    s6, s1, -32
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
    lw      t1, 32(sp)
    addi    s3, x0, 3
    sub     s3, s3, s2
    sub     s3, s3, t1
display_move:  
    # handle peg name
    la      s4, data_peg
    la      s7, data_disk
    add     t0, s4, s2
    add     t2, s4, s3
    # Print "Move Disk "
    la      a1, str1
    li      a2, 10
    li      a7, 0x40
    li      a0, 0x1
    ecall
    # handle & Print disk name
    addi    s6, s1, -32
    add     a1, s7, s6
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
    add     a1, s7, 16
    li      a2, 1
    li      a7, 0x40
    li      a0, 0x1
    ecall

    add     t0, sp, s1
    sw      s3, 0(t0)
    addi    s0, s0, 1
    j       game_loop
finish_game:
    lw      s0, 0(sp)
    lw      s1, 4(sp)
    lw      s2, 8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    lw      s7, 28(sp)
    addi    sp, sp, 56
    ret
.size hanoi,.-hanoi


