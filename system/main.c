#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define printstr(ptr, length)                   \
    do {                                        \
        asm volatile(                           \
            "add a7, x0, 0x40;"                 \
            "add a0, x0, 0x1;" /* stdout */     \
            "add a1, x0, %0;"                   \
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

/* ============= Test Suite ============= */
static void test_my_bfloat16(void)
{

    TEST_LOGGER("Test: My bfloat16\n");
    
    f32_t in1, in2, out,rt;
    in1.value = 0.3f;
    in2.value = 0.5f;
    bf16_t in1_bf, in2_bf, out_bf;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);

    /* Addition */
    out_bf.bits=my_add(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15);
    out.bits=my_add(in1.bits, in2.bits, 0, 9, 23, 31);
    rt.bits = bf16_to_f32(out_bf.bits);
    print_hex(out_bf.bits);
    print_hex(out.bits);
    print_hex(rt.bits);

    /* subtraction */
    out_bf.bits=my_sub(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15);
    out.bits=my_sub(in1.bits, in2.bits, 0, 9, 23, 31);
    rt.bits = bf16_to_f32(out_bf.bits);
    print_hex(out_bf.bits);
    print_hex(out.bits);
    print_hex(rt.bits);
    
    /* Inf checks */
    in1.bits = 0x7f800000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_inf(in1_bf.bits, 0, 0, 25, 7));
    in1.bits = 0x7fc00000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_inf(in1_bf.bits, 0, 0, 25, 7));

    /* NaN checks */
    in1.bits = 0x7fc00000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_nan(in1_bf.bits, 0, 0, 25, 7));
    in1.bits = 0x7f800000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_nan(in1_bf.bits, 0, 0, 25, 7));

    /* Zero checks */
    in1.bits = 0x00000000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_zero(in1_bf.bits, 0, 0, 17));
    in1.bits = 0x80000000;
    in1_bf.bits = f32_to_bf16(in1.bits);
    print_dec(is_zero(in1_bf.bits, 0, 0, 17));

    /* Equality checks */
    in1.value = 3.14159f;
    in2.value = 3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_eq(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));
    in2.value = 3.0f;
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_eq(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));

    /* less than */
    in1.value = 3.0f;
    in2.value = 3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_lt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));
    in1.value = 0.14159f;
    in2.value = -1.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_lt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));

    /* Greater than */
    in1.value = 0.0f;
    in2.value = -3.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_gt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));
    in1.value = 0.14159f;
    in2.value = 1.14159f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    print_dec(is_gt(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15));

    /* Multiplication */
    in1.bits = 62;
    in2.bits = 107;
    out.bits = my_mul(in1.bits, in2.bits, 0, 25, 7, 15, 15);
    print_dec(out.bits);

    /* Floating point Multiplication */
    in1.value = 3.0f;
    in2.value = 5.5f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    out_bf.bits = my_fp_mul(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15, 15);
    out.bits = bf16_to_f32(out_bf.bits);
    print_hex(out.bits);

    /* Floating point Division */
    in1.value = 3.0f;
    in2.value = 5.5f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    in2_bf.bits = f32_to_bf16(in2.bits);
    out_bf.bits = my_div(in1_bf.bits, in2_bf.bits, 0, 25, 7, 15, 15);
    out.bits = bf16_to_f32(out_bf.bits);
    print_hex(out.bits);

    /* Floating point Square Root */
    in1.value = 1.44f;
    in1_bf.bits = f32_to_bf16(in1.bits);
    out_bf.bits = my_sqrt(in1_bf.bits);
    out.bits = bf16_to_f32(out_bf.bits);
    print_hex(out.bits);
}


int main(void)
{
    test_my_bfloat16();
    return 0;
}