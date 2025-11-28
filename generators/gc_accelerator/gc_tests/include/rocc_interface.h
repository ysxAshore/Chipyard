// RISCV Custom Inst Func
#define CUSTOM0 0x0B
#define CUSTOM1 0x2B
#define CUSTOM2 0x5B
#define CUSTOM3 0x7B

#define GET_VALUE1(x) #x
#define GET_VALUE(x) GET_VALUE1(x)

// 构建 riscv 指令
// 31~25 24~20 19~15 14  13  12 11~7  6~0
// func   rs2   rs1  xd xs1 xs2  rd  opcode
#define INST_BIT(opcode, rd, xs2, xs1, xd, rs1, rs2, func)  \
    opcode          | \
    (rd   <<(7))    | \
    (xs2  <<(12))   | \
    (xs1  <<(13))   | \
    (xd   <<(14))   | \
    (rs1  <<(15))   | \
    (rs2  <<(20))   | \
    (func <<(25))

// 三个寄存器操作数 执行结果写rd
#define INS_RRR(rd, rs1, rs2, func)                                           \
 {                                                                            \
     __asm__ __volatile__ (                                                   \
        "sd t0, -24(sp)\n\t"                                                  \
        "sd t1, -16(sp)\n\t"                                                  \
        "sd t2,  -8(sp)\n\t"                                                  \
        "add t1, zero, %1\n\t"                                                \
        "add t2, zero, %2\n\t"                                                \
        ".word " GET_VALUE(INST_BIT(CUSTOM2, 5, 1, 1, 1, 6, 7, func)) "\n\t"  \
        "add %0, zero, t0\n\t"                                                \
        "ld t0, -24(sp)\n\t"                                                  \
        "ld t1, -16(sp)\n\t"                                                  \
        "ld t2,  -8(sp)\n\t"                                                  \
        :"=r"(rd)                                                             \
        :"r" (rs1) , "r" (rs2)                                                \
        );                                                                    \
}

// 两个个寄存器操作数
#define INS_XRR(rs1, rs2, func)                                               \
 {                                                                            \
     __asm__ __volatile__ (                                                   \
        "sd t1, -16(sp)\n\t"                                                  \
        "sd t2,  -8(sp)\n\t"                                                  \
        "add t1, zero, %0\n\t"                                                \
        "add t2, zero, %1\n\t"                                                \
        ".word " GET_VALUE(INST_BIT(CUSTOM2, 5, 1, 1, 0, 6, 7, func)) "\n\t"  \
        "ld t1, -16(sp)\n\t"                                                  \
        "ld t2,  -8(sp)\n\t"                                                  \
        :                                                                     \
        :"r" (rs1) , "r" (rs2)                                                \
        );                                                                    \
}
