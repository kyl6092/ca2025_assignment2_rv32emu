#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define printstr(ptr, length)                   \
    do {                                        \
        asm volatile(                           \
            "li a7, 0x40;"                 \
            "li a0, 0x1;" /* stdout */     \
            "mv a1, %0;"                   \
            "mv a2, %1;" /* length character */ \
            "ecall;"                            \
            :                                   \
            : "r"(ptr), "r"(length)             \
            : "a0", "a1", "a2", "a7");          \
    } while (0)

#define TEST_OUTPUT(msg, length) printstr(msg, length)

#define TEST_LOGGER(msg)                     \
    {                                        \
        char _msg[] = msg;                   \
        TEST_OUTPUT(_msg, sizeof(_msg) - 1); \
    }

extern uint64_t get_cycles(void);
extern uint64_t get_instret(void);

/* Bare metal memcpy implementation */
void *memcpy(void *dest, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *) dest;
    const uint8_t *s = (const uint8_t *) src;
    while (n--)
        *d++ = *s++;
    return dest;
}

/* Software division for RV32I (no M extension) */
static unsigned long udiv(unsigned long dividend, unsigned long divisor)
{
    if (divisor == 0)
        return 0;

    unsigned long quotient = 0;
    unsigned long remainder = 0;

    for (int i = 31; i >= 0; i--) {
        remainder <<= 1;
        remainder |= (dividend >> i) & 1;

        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= (1UL << i);
        }
    }

    return quotient;
}

static unsigned long umod(unsigned long dividend, unsigned long divisor)
{
    if (divisor == 0)
        return 0;

    unsigned long remainder = 0;

    for (int i = 31; i >= 0; i--) {
        remainder <<= 1;
        remainder |= (dividend >> i) & 1;

        if (remainder >= divisor) {
            remainder -= divisor;
        }
    }

    return remainder;
}

/* Software multiplication for RV32I (no M extension) */
static uint32_t umul(uint32_t a, uint32_t b)
{
    uint32_t result = 0;
    while (b) {
        if (b & 1)
            result += a;
        a <<= 1;
        b >>= 1;
    }
    return result;
}

/* Provide __mulsi3 for GCC */
uint32_t __mulsi3(uint32_t a, uint32_t b)
{
    return umul(a, b);
}

/* Simple integer to hex string conversion */
static void print_hex(unsigned long val)
{
    char buf[20];
    char *p = buf + sizeof(buf) - 1;
    *p = '\n';
    p--;

    if (val == 0) {
        *p = '0';
        p--;
    } else {
        while (val > 0) {
            int digit = val & 0xf;
            *p = (digit < 10) ? ('0' + digit) : ('a' + digit - 10);
            p--;
            val >>= 4;
        }
    }

    p++;
    printstr(p, (buf + sizeof(buf) - p));
}

/* Simple integer to decimal string conversion */
static void print_dec(unsigned long val)
{
    char buf[20];
    char *p = buf + sizeof(buf) - 1;
    *p = '\n';
    p--;

    if (val == 0) {
        *p = '0';
        p--;
    } else {
        while (val > 0) {
            *p = '0' + umod(val, 10);
            p--;
            val = udiv(val, 10);
        }
    }

    p++;
    printstr(p, (buf + sizeof(buf) - p));
}

/* ============= BFloat16 Implementation ============= */

typedef struct {
    uint16_t bits;
} bf16_t;
typedef union f32{
    uint32_t bits;
    float value;
}f32_t;

/* ============= my bfloat16 Declaration ============= */

extern uint16_t f32_to_bf16(const uint32_t in);
extern uint32_t bf16_to_f32(const uint16_t in);

extern bool is_inf(
    const uint32_t in,
    const uint32_t reserv1,
    const uint32_t reserv2,
    const uint32_t mant_offset,
    const uint32_t exp_offset
);
extern bool is_nan(
    const uint32_t in,
    const uint32_t reserv1,
    const uint32_t reserv2,
    const uint32_t mant_offset,
    const uint32_t exp_offset
);
extern bool is_zero(
    const uint32_t in,
    const uint32_t reserv1,
    const uint32_t reserv2,
    const uint32_t mant_exp_offset
);

extern bool is_eq(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset
);

extern bool is_lt(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset
);

extern bool is_gt(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset
);

extern uint32_t my_add(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset
);

extern uint32_t my_sub(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset
);

extern uint32_t my_mul(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset,
    const uint32_t oper_offset
);

extern uint32_t my_fp_mul(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset,
    const uint32_t oper_offset
);

extern uint32_t my_div(
    const uint32_t in1,
    const uint32_t in2,
    const uint32_t reserv,
    const uint32_t mant_offset,
    const uint32_t exp_offset,
    const uint32_t sign_offset,
    const uint32_t oper_offset
);

extern uint32_t my_sqrt(
    const uint32_t in
);

extern void hanoi();

/* ============= Test Suite ============= */
#define BF16_NAN() ((bf16_t) {.bits = 0x7FC0})
#define BF16_ZERO() ((bf16_t) {.bits = 0x0000})
#define BF16_INF() ((bf16_t) {.bits = 0x7f80})

static void test_bf16_add(void)
{
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf, out_bf, expect;
    in1.value = 0.3f;
    in2.value = 0.5f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    expect.bits = 0x3f4d;
    out_bf.bits=my_add(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15);

    /* Optional (Test with FP32) */
    // f32_t out, rt;
    // out.bits=my_add(in1.bits, in2.bits, 0, 9, 23, 31);
    // rt.bits = bf16_to_f32(out_bf.bits);

    /* Check Correctness */
    if (out_bf.bits == expect.bits) {
        TEST_LOGGER("bf16 FP Addition \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP Addition \t\tFAILED (expected 0x3f4d)\n");
    }
}

static void test_bf16_sub(void)
{
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf, out_bf, expect;
    in1.value = 0.3f;
    in2.value = 0.5f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    expect.bits = 0xbe4c;
    out_bf.bits=my_sub(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15);

    /* Optional (Test with FP32) */
    // f32_t out, rt;
    // out.bits=my_sub(in1.bits, in2.bits, 0, 9, 23, 31);
    // rt.bits = bf16_to_f32(out_bf.bits);

    /* Check Correctness */
    if (out_bf.bits == expect.bits) {
        TEST_LOGGER("bf16 FP Subtraction \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP Subtraction \t\tFAILED (expected 0xbe4c)\n");
    }
}

static void test_bf16_mul(void)
{
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf, out_bf, expect;
    in1.value = 2.0f;
    in2.value = 3.0f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    expect.bits = 0x40c0;
    out_bf.bits = my_fp_mul(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15, 15);
    /* Optional (Test with FP32) */
    // f32_t out, rt;
    // out.bits=my_mul(in1.bits, in2.bits, 0, 9, 23, 31, 48);
    // rt.bits = bf16_to_f32(out_bf.bits);

    /* Check Correctness */
    if (out_bf.bits == expect.bits) {
        TEST_LOGGER("bf16 FP Multiplication \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP Multiplication \t\tFAILED (expected 0x4184)\n");
    }
}

static void test_bf16_div(void) {
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf, out_bf, expect;
    in1.value = 3.0f;
    in2.value = 5.5f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    expect.bits = 0x3f0b;
    out_bf.bits = my_div(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15, 15);

    /* Optional (Test with FP32) */
    // f32_t out, rt;
    // out.bits=my_div(in1.bits, in2.bits, 0, 9, 23, 31, 48);
    // rt.bits = bf16_to_f32(out_bf.bits);

    /* Check Correctness */
    if (out_bf.bits == expect.bits) {
        TEST_LOGGER("bf16 FP Division \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP Division \t\tFAILED (expected 0x3f0b)\n");
    }
}

static void test_bf16_sqrt(void) {
    /* Kernel Function */
    f32_t in1;
    bf16_t in1_bf, out_bf, expect;
    in1.value = 1.44f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    expect.bits = 0x3f99;
    out_bf.bits = my_sqrt(in1_bf.bits);
    /* Check Correctness */
    if (out_bf.bits == expect.bits) {
        TEST_LOGGER("bf16 FP Square Root \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP Square Root \t\tFAILED (expected 0x3f0b)\n");
    }
}

static void test_bf16_NaN(void) {

    bf16_t in_bf = BF16_NAN();
    if (is_nan(in_bf.bits, 0, 0, 25, 7)) {
        TEST_LOGGER("bf16 FP NaN Checks \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP NaN Checks \t\tFAILED\n");
    }
}

static void test_bf16_INF(void) {
    bf16_t in_bf = BF16_INF();
    if (is_inf(in_bf.bits, 0, 0, 25, 7)) {
        TEST_LOGGER("bf16 FP +INF Checks \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP +INF Checks \t\tFAILED\n");
    }
}

static void test_bf16_ZERO(void) {
    bf16_t in_bf = BF16_ZERO();
    if (is_zero(in_bf.bits, 0, 0, 17)) {
        TEST_LOGGER("bf16 FP +0 Checks \t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP +0 Checks \t\tFAILED\n");
    }
}

static void test_bf16_EQUAL(void) {
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf;
    in1.value = 3.14159f;
    in2.value = 3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);

    /* Check Correctness */
    if (is_eq(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15)) {
        in2.value = 3.0f;
        in2_bf.bits = f32_to_bf16(in2.bits);
        if (!is_eq(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15)) {
            TEST_LOGGER("bf16 FP Equality \t\tPASSED\n");
        }
        else {
            TEST_LOGGER("bf16 FP Equality \t\tFAILED\n");
        }
    }
    else {
        TEST_LOGGER("bf16 FP Equality \t\tFAILED\n");
    }
}

static void test_bf16_LT(void) {
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf;
    in1.value = 3.0f;
    in2.value = 3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);

    /* Check Correctness */
    if (is_lt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15)) {
        TEST_LOGGER("bf16 FP LT \t\t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP LT \t\t\tFAILED\n");
    }
}

static void test_bf16_GT(void) {
    /* Kernel Function */
    f32_t in1, in2;
    bf16_t in1_bf, in2_bf;
    in1.value = 0.0f;
    in2.value = -3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);

    /* Check Correctness */
    if (is_gt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15)) {
        TEST_LOGGER("bf16 FP GT \t\t\tPASSED\n");
    }
    else {
        TEST_LOGGER("bf16 FP GT \t\t\tFAILED\n");
    }
}

static void test_my_bfloat16(void)
{
    uint64_t start_cycles, end_cycles, cycles_elapsed;
    uint64_t start_instret, end_instret, instret_elapsed;

    TEST_LOGGER("--------------------\n");
    TEST_LOGGER("Test: My bfloat16\n");
    
    f32_t in1, in2, out, rt;
    bf16_t in1_bf, in2_bf, out_bf, expect;

    

    /* Addition */
    start_cycles = get_cycles();
    start_instret = get_instret();
    
    test_bf16_add();
          
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");
    
    /* Subtraction */
    start_cycles = get_cycles();
    start_instret = get_instret();
    
    test_bf16_sub();
          
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");
    
    /* Floating point Multiplication */
    start_cycles = get_cycles();
    start_instret = get_instret();
    
    test_bf16_mul();
          
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Floating point Division */
    start_cycles = get_cycles();
    start_instret = get_instret();
    
    test_bf16_div();
          
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Floating point Square Root */
    start_cycles = get_cycles();
    start_instret = get_instret();
    
    test_bf16_sqrt();
          
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Floating point Specail cases */

    /* NaN checks */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_NaN();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Inf checks */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_INF();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Zero checks */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_ZERO();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");   

    /* Equality checks */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_EQUAL();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");
    

    /* less than */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_LT();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

    /* Greater than */
    start_cycles = get_cycles();
    start_instret = get_instret();

    test_bf16_GT();

    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");

}

static void test_hanoi(void)
{
    uint64_t start_cycles, end_cycles, cycles_elapsed;
    uint64_t start_instret, end_instret, instret_elapsed;
    TEST_LOGGER("--------------------\n");
    TEST_LOGGER("Test: My hanoi\n");

    start_cycles = get_cycles();
    start_instret = get_instret();
    hanoi();
    end_cycles = get_cycles();
    end_instret = get_instret();
    cycles_elapsed = end_cycles - start_cycles;
    instret_elapsed = end_instret - start_instret;

    TEST_LOGGER("  Cycles: ");
    print_dec((unsigned long) cycles_elapsed);
    TEST_LOGGER("  Instructions: ");
    print_dec((unsigned long) instret_elapsed);
    TEST_LOGGER("\n");
}
int main(void)
{
    test_my_bfloat16();
    test_hanoi();
    return 0;
}