#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include "encoding.h"
#include "marchid.h"
#include "cuteMarcoinstHelper.h"

#include "conv_48.h"

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

#define HAS_MVIN_SCALE
typedef float scale_t;
typedef uint32_t scale_t_bits;

typedef int32_t scale_acc_t;
typedef uint32_t scale_acc_t_bits;

typedef float acc_scale_t;
typedef uint32_t acc_scale_t_bits;

#define CUTE_TILE_Tensor_M 64
#define CUTE_TILE_Tensor_N 64
#define CUTE_TILE_Tensor_K 64

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#define CUTE_INT8 1

void CUTE_CONV_3_3_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);//完成二维张量的切分，确定CUTE任务的切分
void CUTE_CONV_3_3_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);
void CUTE_CONV_1_1_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);
void CUTE_CONV_1_1_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);

// int cute_temp[4*4096*256];

uint64_t CUTE_CONV_3_3_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
uint64_t CUTE_CONV_3_3_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
uint64_t CUTE_CONV_1_1_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
uint64_t CUTE_CONV_1_1_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
uint64_t CUTE_CONV_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);

uint64_t CUTE_CONV_KERNEL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);


uint64_t CUTE_MATMUL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);//64*64*64

char *activation_name(int act) {
  switch (act) {
    case NO_ACTIVATION:
      return "NO_ACTIVATION";
    case RELU:
      return "RELU";
    case LAYERNORM:
      return "LAYERNORM";
    case IGELU:
      return "IGELU";
    case SOFTMAX:
      return "SOFTMAX";
    default:
      return "UNKNOWN";
  }
}

uint64_t vec_time = 0;
// 512位对齐的数组acc_t result[64][64]
acc_t __attribute__((aligned(64))) CUTE_result[2][64][3072];//double buffer

int CUTE_result_index = 0;

static uint64_t read_cycles() {
    
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;

}


static void resadd_cpu(const size_t I, const size_t J,
        const int A_scale,
        const int B_scale,
        const int C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}

//宏展开，更多的高性能算子，额外的指令段
static void resadd_cpu_greater(const size_t I, const size_t J,
        const int A_scale,
        const int B_scale,
        const int C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE_GREATER(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}


void scale_after_operation_64_64(acc_t input[64][64], int dim_i,int dim_j,elem_t * output,int scale_shift,uint64_t stride_c)
{

    //normal scale
    // // printf("normal scale x:");
    // for (size_t i = 0; i < dim_i; i++) {
    //     for (size_t j = 0; j < dim_j; j++) {
    //         elem_t* c = output + i * stride_c + j;
    //         acc_t x = input[i][j];
    //         // if(dim_i !=64)
    //         // printf("%d ",x);
    //         x = ACC_SCALE(x, scale_shift);
    //         *c = x;
    //     }
    // }

    //vector scale
    uint64_t start = read_cycles();
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j+=8) {
            for (size_t k = 0; k < 8; k++) {
                elem_t* c = output + i * stride_c + j+k;
                acc_t x = input[i][j+k];
                x = x >> scale_shift;
                // Clip result
                x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
                *c = x;
            }
        }
    }
    uint64_t end = read_cycles();
    vec_time += end - start;

}

void scale_after_operation_64_64_relu(acc_t input[64][64], int dim_i,int dim_j,elem_t * output,int scale_shift,uint64_t stride_c)
{

    //relu scale
    // for (size_t i = 0; i < dim_i; i++) {
    //     for (size_t j = 0; j < dim_j; j++) {
    //         elem_t* c = output + i * stride_c + j;
    //         acc_t x = input[i][j];
    //         // printf("%d ",x);
    //         x = ACC_SCALE(x, scale_shift);
    //         *c = x < 0 ? 0 : x;;
    //     }
    // }
    uint64_t start = read_cycles();
    //vector scale
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j+=8) {
            for (size_t k = 0; k < 8; k++) {
                elem_t* c = output + i * stride_c + j+k;
                acc_t x = input[i][j+k];
                x = x >> scale_shift;
                x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
                *c = x;
            }
        }
    }
    uint64_t end = read_cycles();
    vec_time += end - start;
}

void scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_relu_shift_x(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I)
{
    //shift 8 是可以有特殊优化的，只load 7bit+1bit，就能完成relu和shift
    __asm__ volatile (
        // 初始化寄存器
        "mv a3, %[input]            \n" // a3 = input 起始地址
        "mv a4, %[output]           \n" // a4 = output 起始地址
        "mv a5, %[stride_input]     \n" // a5 = input 的行步长
        "mv a6, %[stride_output]    \n" // a6 = output 的行步长
        "mv a7, %[shift_scale]      \n" // a7 = shift_scale
        
        "mv t1, %[dim_I]            \n" // 行计数器 t1 = 64
        "li t2, 127                 \n" // relu_max = 127
        "li t3, 0                   \n" // input列寄存器 a3 + 32
        "li t4, 0                   \n" // input列寄存器 a3 + 64
        "li t5, 0                   \n" // input列寄存器 a3 + 96
        "li t6, 0                   \n" // input列寄存器 a3 + 128
        "li s2, 0                   \n" // input列寄存器 a3 + 160
        "li s3, 0                   \n" // input列寄存器 a3 + 192
        "li s4, 0                   \n" // input列寄存器 a3 + 224
        "li t0, 0                   \n" // output列寄存器 a4 + 32

        // 设置向量长度，假设 vlen 为 256 位（32 个 8-bit 元素）

        "1:                         \n" // 外层循环标签 row_loop

        // 加载数据，避免load指令重复依赖
        "addi t3, a3, 32             \n" // t3 = a3 + 32
        "addi t4, a3, 64             \n" // t4 = a3 + 64
        "addi t5, a3, 96             \n" // t5 = a3 + 96
        "addi t6, a3, 128            \n" // t6 = a3 + 128
        "addi s2, a3, 160            \n" // s2 = a3 + 160
        "addi s3, a3, 192            \n" // s3 = a3 + 192
        "addi s4, a3, 224            \n" // s4 = a3 + 224

        "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vle32.v v1, (a3)            \n" // 加载 input 的前 8 个元素到 v1
        "vle32.v v2, (t3)            \n" // 加载 input 的下 8 个元素到 v2
        "vle32.v v3, (t4)            \n" // 加载 input 的下 8 个元素到 v3
        "vle32.v v4, (t5)            \n" // 加载 input 的下 8 个元素到 v4
        "vle32.v v5, (t6)            \n" // 加载 input 的下 8 个元素到 v5
        "vle32.v v6, (s2)            \n" // 加载 input 的下 8 个元素到 v6
        "vle32.v v7, (s3)            \n" // 加载 input 的下 8 个元素到 v7
        "vle32.v v8, (s4)            \n" // 加载 input 的下 8 个元素到 v8

        // 向量操作，ReLU 和移位
        "vsetvli t0, zero, e16, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vnsra.wx v9, v1,  a7          \n"  // v9  = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v10, v3, a7          \n" // v10 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v11, v5, a7          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v12, v7, a7          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vminu.vx v9, v9, t2       \n"   // v1 = minu(v1, 127) // ReLU 操作,负数的补码是大于127的，所以直接一条vmaxu即可
        "vminu.vx v10, v10, t2       \n" // v2 = minu(v2, 127)
        "vminu.vx v11, v11, t2       \n" // v3 = minu(v3, 127)
        "vminu.vx v12, v12, t2       \n" // v4 = minu(v4, 127)

        "vsetvli t0, zero, e8, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次，缩小到8bit
        "vnsra.wi v13, v9, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wi v14, v11, 0          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))

        // 将数据转换为 8-bit，并存储到v9-v10
        "addi t0, a4, 32            \n"  // output的行下一个起始地址

        // 存储数据（连续写回）
        "vse8.v v13, (a4)       \n" // 存储 v20 到 output（32 个元素）
        "vse8.v v14, (t0)       \n" // 存储 v21 到 output（32 个元素）

        // 更新行指针
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a4, a4, a6             \n" // a4 前进 stride_c 字节（下一行的起始位置）
        "addi t1, t1, -1            \n" // 行计数器 t1--
        "bnez t1, 1b                \n" // 如果 t1 != 0，跳转到 row_loop

        : // 输出寄存器（空）
        : [input] "r" (input), [output] "r" (output), [stride_input] "r" (stride_input), [stride_output] "r" (stride_output), [shift_scale] "r" (shift_scale), [dim_I] "r" (dim_I)  // 输入约束
        : "t0", "t1","t2","t3","t4","t5","t6","s2","s3","s4", "a3", "a4", "a5","a6", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "memory" // 破坏描述符
    );
}

void scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I)
{
    //shift 8 是可以有特殊优化的，只load 7bit+1bit，就能完成relu和shift
    __asm__ volatile (
        // 初始化寄存器
        "mv a3, %[input]            \n" // a3 = input 起始地址
        "mv a4, %[output]           \n" // a4 = output 起始地址
        "mv a5, %[stride_input]     \n" // a5 = input 的行步长
        "mv a6, %[stride_output]    \n" // a6 = output 的行步长
        "mv a7, %[shift_scale]      \n" // a7 = shift_scale
        
        "mv t1, %[dim_I]            \n" // 行计数器 t1 = 64
        "li t2, 127                 \n" // relu_max = 127
        "li t3, 0                   \n" // input列寄存器 a3 + 32
        "li t4, 0                   \n" // input列寄存器 a3 + 64
        "li t5, 0                   \n" // input列寄存器 a3 + 96
        "li t6, 0                   \n" // input列寄存器 a3 + 128
        "li s2, 0                   \n" // input列寄存器 a3 + 160
        "li s3, 0                   \n" // input列寄存器 a3 + 192
        "li s4, 0                   \n" // input列寄存器 a3 + 224
        "li t0, 0                   \n" // output列寄存器 a4 + 32

        // 设置向量长度，假设 vlen 为 256 位（32 个 8-bit 元素）

        "1:                         \n" // 外层循环标签 row_loop

        // 加载数据，避免load指令重复依赖
        "addi t3, a3, 32             \n" // t3 = a3 + 32
        "addi t4, a3, 64             \n" // t4 = a3 + 64
        "addi t5, a3, 96             \n" // t5 = a3 + 96
        "addi t6, a3, 128            \n" // t6 = a3 + 128
        "addi s2, a3, 160            \n" // s2 = a3 + 160
        "addi s3, a3, 192            \n" // s3 = a3 + 192
        "addi s4, a3, 224            \n" // s4 = a3 + 224

        "vsetvli t0, zero, e32, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vle32.v v1, (a3)            \n" // 加载 input 的前 8 个元素到 v1
        "vle32.v v2, (t3)            \n" // 加载 input 的下 8 个元素到 v2
        "vle32.v v3, (t4)            \n" // 加载 input 的下 8 个元素到 v3
        "vle32.v v4, (t5)            \n" // 加载 input 的下 8 个元素到 v4
        "vle32.v v5, (t6)            \n" // 加载 input 的下 8 个元素到 v5
        "vle32.v v6, (s2)            \n" // 加载 input 的下 8 个元素到 v6
        "vle32.v v7, (s3)            \n" // 加载 input 的下 8 个元素到 v7
        "vle32.v v8, (s4)            \n" // 加载 input 的下 8 个元素到 v8

        // 向量操作，ReLU 和移位
        "vsetvli t0, zero, e16, m1   \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次,先缩小到16bit
        "vnsra.wx v9,  v1, a7          \n"  // v9  = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v10, v3, a7          \n" // v10 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v11, v5, a7          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wx v12, v7, a7          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))

        "vmin.vx v9, v9, t2       \n"   // v1 = min(v1, 127) 
        "vmin.vx v10, v10, t2       \n" // v2 = min(v2, 127)
        "vmin.vx v11, v11, t2       \n" // v3 = min(v3, 127)
        "vmin.vx v12, v12, t2       \n" // v4 = min(v4, 127)

        "vmax.vx v9, v9, t2       \n"   // v1 = max(v1, -128)
        "vmax.vx v10, v10, t2       \n" // v2 = max(v2, -128)
        "vmax.vx v11, v11, t2       \n" // v3 = max(v3, -128)
        "vmax.vx v12, v12, t2       \n" // v4 = max(v4, -128)

        "vsetvli t0, zero, e8, m1  \n" // 设置每个向量寄存器宽度为 256 位（32 x 8-bit 元素）由于没有一口气sew缩小4倍的指令，所以我们要缩小2次，缩小到8bit
        "vnsra.wi v13, v9, 0          \n" // v11 = cat (clip(v1 >> 8), clip(v2 >> 8))
        "vnsra.wi v14, v11, 0          \n" // v12 = cat (clip(v1 >> 8), clip(v2 >> 8))

        // 将数据转换为 8-bit，并存储到v9-v10
        "addi t0, a4, 32            \n"  // output的行下一个起始地址

        // 存储数据（连续写回）
        "vse8.v v13, (a4)       \n" // 存储 v20 到 output（32 个元素）
        "vse8.v v14, (t0)       \n" // 存储 v21 到 output（32 个元素）

        // 更新行指针
        "add a3, a3, a5             \n" // a3 前进 32 字节（下一行的起始位置）
        "add a4, a4, a6             \n" // a4 前进 stride_c 字节（下一行的起始位置）
        "addi t1, t1, -1            \n" // 行计数器 t1--
        "bnez t1, 1b                \n" // 如果 t1 != 0，跳转到 row_loop

        : // 输出寄存器（空）
        : [input] "r" (input), [output] "r" (output), [stride_input] "r" (stride_input), [stride_output] "r" (stride_output), [shift_scale] "r" (shift_scale), [dim_I] "r" (dim_I)  // 输入约束
        : "t0", "t1","t2","t3","t4","t5","t6","s2","s3","s4", "a3", "a4", "a5","a6", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "memory" // 破坏描述符
    );
}


static void matmul_cute(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias,int transpose_result) {

// scale = 1.0;
  const int no_bias = D == NULL;
  //输出所有输入参数
    // printf("transA:%d,transB:%d,DIM_I:%d,DIM_J:%d,DIM_K:%d,stride_A:%d,stride_B:%d,stride_D:%d,stride_C:%d\nA_scale_factor:%f,B_scale_factor:%f,D_scale_factor:%d,act:%d,scale:%f,bert_scale:%f,repeating_bias:%d\n",transA,transB,DIM_I,DIM_J,DIM_K,stride_A,stride_B,stride_D,stride_C,A_scale_factor,B_scale_factor,D_scale_factor,act,scale,bert_scale,repeating_bias);
  //如果不是layernorm或者softmax切成64,64,K的小块，然后每次完成计算，调用向量算子。
  //如果是layernorm或者softmax，或者不是64,M,K的小块，然后调用向量算子。

  if(!(DIM_I % 64 == 0 && DIM_J % 64 == 0 && DIM_K % 64 == 0))
  {
    printf("Can't Till Now!");
    //TODO:添加部分矩阵乘算子
    exit(1);
  }

  if(DIM_J > 3072 && (act == LAYERNORM || act == SOFTMAX))
  {
    printf("DIM_J too large!");
    exit(1);
  }

  void (*afater_operation)(acc_t *,int,int,elem_t *,acc_scale_t,int) = NULL;

  switch (act) {
    case NO_ACTIVATION:
      afater_operation = scale_after_operation_64_64;
      break;
    case RELU:
      afater_operation = scale_after_operation_64_64;
      break;
    default:
      afater_operation = scale_after_operation_64_64;
      break;
  }
  
//   printf("!!\n[matmul_cute] START!!\n!!\n");
  if(act != LAYERNORM && act != SOFTMAX)
  {
    int Tile_I = DIM_I / 64;
    int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = 64;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = stride_C;
    int Application_stride_D = stride_D;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = repeating_bias;
    int Is_Zero_Load = no_bias;

    elem_t* Tile_A = A;
    elem_t* Tile_B = B;
    acc_t* Tile_C = CUTE_result[CUTE_result_index];
    acc_t* Tile_D = D;


    //后操作的函数指针，返回值是void
    
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */

    int i = 0;
    int j = 1;
    int pre_i = 0;
    int pre_j = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
    for (i=0;i<Tile_I;i++)
    for (j=(i==0?1:0);j<Tile_J;j++)
    {
        //等待CUTE任务完成
        // while(acc_not_finish)
        // {
        //     /*
        //     cute 完成查询
        //     */
        //    //假查询
        // }

        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*Tile_J+j,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A + j * 64;
        Tile_B = B + i * 64 * stride_B + j * 64;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = D;
        /*
        cute 配置
        cute 指令发射
        */
        
        //执行当前任务的CPU的向量后操作任务
        // printf("pre_i:%d,pre_j:%d\n",pre_i,pre_j);
        afater_operation(CUTE_result[CUTE_result_index],64,64,(C+pre_i*64*stride_C+pre_j*64),scale,stride_C);
        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*DIM_J+j,DIM_K);
        // printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
        pre_j = j;
    }
    // printf("[Final]pre_i:%d,pre_j:%d\n",pre_i,pre_j);
    afater_operation(CUTE_result[CUTE_result_index],64,64,(C+pre_i*64*stride_C+pre_j*64),scale,stride_C);
    // printf("[Final][Vec]Vector Operation %s Finish\n",activation_name(act));
    

  }else
  {
    int Tile_I = DIM_I / 64;
    // int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = 64;
    int Application_K = DIM_K;

    int Application_stride_A = stride_A;
    int Application_stride_B = stride_B;
    int Application_stride_C = stride_C;
    int Application_stride_D = stride_D;

    int Is_Transpose = transpose_result;
    int Is_repeating_row = repeating_bias;
    int Is_Zero_Load = no_bias;

    elem_t* Tile_A = A;
    elem_t* Tile_B = B;
    acc_t * Tile_C = CUTE_result[CUTE_result_index];
    acc_t * Tile_D = D;


    //后操作的函数指针，返回值是void
    // void (*afater_operation)(acc_t *,int,int,elem_t *,acc_scale_t,int) = NULL;
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */

    int i = 0;
    int pre_i = 0;

    int acc_not_finish = 1;
    for (i=0;i<Tile_I;i++)
    {
        //等待CUTE任务完成
        // while(acc_not_finish)
        // {
        //     /*
        //     cute 完成查询
        //     */
        //    //假查询
        // }

        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A ;
        Tile_B = B + i * 64 * stride_B ;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = D;
        /*
        cute 配置
        cute 指令发射
        */
        
        //执行当前任务的CPU的向量后操作任务
        // printf("pre_i:%d\n",pre_i);
        afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
        // printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }

    afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
    
  }
}


void CUTE_MATMUL_MarcoTask_SIM(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,void * VectorOp,int VectorInst_Length)
{
    elem_t * Tile_A = (elem_t *)A;
    elem_t * Tile_B = (elem_t *)B;
    acc_t * Tile_C = (acc_t *)C;
    acc_t * Bias_D = (acc_t *)D;

    acc_t bias_row[64] = {0};
    for(int i=0;i<64;i++){
        bias_row[i] = Bias_D[i];
    }

    for(int i=0;i<Application_M;i++){
        for(int j=0;j<Application_N;j++){
            acc_t result = 0;
            for(int k=0;k<Application_K;k++){
                result += Tile_A[i*stride_A+k]*Tile_B[j*stride_B+k];
            }
            Tile_C[i*stride_C/4+j] = result + bias_row[j];
        }
    }

    //输出前1000个元素
    // printf("CUTE_MATMUL_MarcoTask_SIM:");
    // for(int i=0;i<1000;i++){
    //     int dj = i % Application_N;
    //     int di = i / Application_N;

        // printf("[%d]%d ",i,Tile_C[di*stride_D+dj]);
    // }
    // printf("\n");
    // exit(1);
}



void CUTE_CONV_KERNEL_MarcoTask_SIM(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{

    elem_t * Tile_A = (elem_t *)A;
    elem_t * Tile_B = (elem_t *)B;
    acc_t * Tile_C = (acc_t *)C;
    acc_t * Bias_D = (acc_t *)D;

    acc_t bias_row[64] = {0};
    for(int i=0;i<64;i++){
        bias_row[i] = Bias_D[i];
    }
    //输出A的地址
    // printf("Tile_A:%p\n",Tile_A);
    //输出oh和ow
    // printf("conv_oh_index:%d,conv_ow_index:%d\n",conv_oh_index,conv_ow_index);

    // int kernel_position[9][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,0},{0,1},{1,-1},{1,0},{1,1}};

    if(kernel_size %2 != 1)
    {
        printf("kernel_size must be odd\n");
        exit(1);
    }

    acc_t temp_acc[64][64] = {0};
    int p = 0;
    for(int kernel_height = - kernel_size/2;kernel_height<=kernel_size/2;kernel_height++)
    for(int kernel_weight = - kernel_size/2;kernel_weight<=kernel_size/2;kernel_weight++,p++)
    {
        int conv_oh = conv_oh_index;
        int conv_ow = conv_ow_index;
        for(int i=0;i<Application_M;i++)
        {
            int ih_with_kernel = conv_oh * conv_stride + kernel_height;
            int iw_with_kernel = conv_ow * conv_stride + kernel_weight;
            // printf("oh=%d,ow=%d;",conv_oh,conv_ow);
            if ((ih_with_kernel < 0 || ih_with_kernel >= conv_oh_max*conv_stride || iw_with_kernel < 0 || iw_with_kernel >= conv_ow_max*conv_stride))
            {
                // printf("[SKIP]ih_with_kernel:%d,iw_with_kernel:%d\n",ih_with_kernel,iw_with_kernel);
            }
            else
            {
                // printf("ih_with_kernel:%d,iw_with_kernel:%d\n",ih_with_kernel,iw_with_kernel);
                for(int j=0;j<Application_N;j++)
                {
                    acc_t result = 0;
                    for(int k=0;k<Application_K;k++)
                    {
                        result += Tile_A[(ih_with_kernel*conv_ow_max*conv_stride+iw_with_kernel)*stride_A+k]*Tile_B[p*kernel_stride+j*stride_B+k];
                    }
                    temp_acc[i][j] += result;
                }
            }
            conv_ow += 1;
            if(conv_ow >= conv_ow_max)
            {
                conv_ow = 0;
                conv_oh += 1;
            }
        }
    }

    for(int i=0;i<Application_M;i++){
        for(int j=0;j<Application_N;j++){
            Tile_C[i*stride_C/4+j] = temp_acc[i][j] + bias_row[j];
        }
    }

    // if(conv_oh_index == 0 && conv_ow_index == 0)
    // {
    //     //输出前1000个
        // printf("CUTE_CONV_MarcoTask_SIM:");
    //     for(int i=0;i<1000;i++){
            // printf("%d ",Tile_C[i]);
    //     }
    // }
}

void CUTE_CONV_MarcoTask_SIM(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    elem_t * Tile_A = (elem_t *)A;
    elem_t * Tile_B = (elem_t *)B;
    acc_t * Tile_C = (acc_t *)C;
    acc_t * Bias_D = (acc_t *)D;

    acc_t bias_row[64] = {0};
    for(int i=0;i<64;i++){
        bias_row[i] = Bias_D[i];
    }
    //输出A的地址
    // printf("Tile_A:%p\n",Tile_A);
    int conv_iw = conv_ow_index*conv_stride;
    int conv_ih = conv_oh_index*conv_stride;
    for(int i=0;i<Application_M;i++){
        // printf("ih:%d,iw:%d\n",conv_ih,conv_iw);
        for(int j=0;j<Application_N;j++){
            acc_t result = 0;
            for(int k=0;k<Application_K;k++){
                result += Tile_A[(conv_ih*conv_ow_max*conv_stride+conv_iw)*stride_A+k]*Tile_B[j*stride_B+k];
            }
            Tile_C[i*stride_C/4+j] = result + bias_row[j];
        }
        conv_iw += conv_stride;
        if(conv_iw >= conv_ow_max*conv_stride){
            conv_iw = 0;
            conv_ih += conv_stride;
        }
    }
    // if(conv_oh_index == 2 && conv_ow_index == 8)
    // {
    //     //输出前1000个
        // printf("CUTE_CONV_MarcoTask_SIM:");
    //     for(int i=0;i<1000;i++){
            // printf("%d ",Tile_C[i]);
    //     }
    // }
}

uint64_t CUTE_MATMUL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    // printf("CUTE_MATMUL_MarcoTask\n");
    // printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    // printf("stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",stride_A,stride_B,stride_C,stride_D);

    //CUTE配置
    //CUTE指令发射

    /*
    注意实现，关于TensorLoad，Application_M可以不满64。
    Application_M不满64时，TileA的Load会提早结束。
    
        由于计算时，Application_M不满64，所以计算也会提早结束，但是计算部件是Matrix_M,Matrix_N,Matrix_K的计算部件，所以每次最少的计算单元是Matrix_M*Matrix_N*Matrix_K，
    此时，生成的结果为Matrix_M*Matrix_N*ResultWitdh的数据，会送入Reorder_FIFO,Reorder_FIFO正常会连续接受{CSP_DATAWIDTH/(Matrxi_N*ResultWidth)}个数据[目的是为了让一个CSP_DATAWIDTH的数据是Channel First的数据],
    然后完成重排序后送入VectorInterface，此时可加一个标志位确定是否需要后操作(填充的值直接不需要后操作,直接提早结束注意int32和int8需要补的拍),如果VectorInterface完成任务，则会输入到CSP_WRITE_FIFO中，CSP_WRITE_FIFO会连续接受数据，直到CSP_WRITE_FIFO填充至CSP_DATAWIDTH[VectroInterface的输出数据可能是量化后的数据所以bit数可能更低]，
    SCP_Write_FIFO根据送入的数据，计算偏移，让SCP内完全Matrix_N主序的数据(根据Matrix_N的大小，可能提前结束)。
    */
    //矩阵乘就是IH=1，IW=M，IC=K，OC=N，KH=1，KW=1，STRIDE=1的卷积

    printf("invalid func[CUTE_MATMUL_MarcoTask]! wait YJP for finish it\n");
    exit(1);
    return -1;

    // return CUTE_CONV_KERNEL_MarcoTask_SIM(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,1,0,\
    //                         stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

    

}

uint64_t CUTE_CONV_KERNEL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    // printf("CUTE_CONV_KERNEL_MarcoTask\n");
    // printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    // printf("stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",stride_A,stride_B,stride_C,stride_D);
    // printf("conv_stride:%d,kernel_size:%d\n",conv_stride,kernel_size);
    // printf("conv_oh_index:%d,conv_ow_index:%d,conv_oh_max:%d,conv_ow_max:%d\n",conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max);

    //CUTE配置
    //CUTE指令发射

    // uint64_t issue_cute_conv_marco_inst(uint64_t ATensor_Base_Addr,uint64_t ATensor_M_Stride,
    //                                    uint64_t BTensor_Base_Addr,uint64_t BTensor_M_Stride,
    //                                    uint64_t CTensor_Base_Addr,uint64_t CTensor_M_Stride,
    //                                    uint64_t DTensor_Base_Addr,uint64_t DTensor_M_Stride,
    //                                    uint64_t M,uint64_t N,uint64_t K,uint64_t kernel_stride,
    //                                    uint64_t element_type,uint64_t bias_type,uint64_t transpose_result,uint64_t conv_stride,uint64_t conv_oh_max,uint64_t conv_ow_max,
    //                                    uint64_t kernel_size,uint64_t conv_oh_per_add,uint64_t conv_ow_per_add,uint64_t conv_oh_index,uint64_t conv_ow_index)

    return 0;

//     return CUTE_CONV_KERNEL_MarcoTask_SIM(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,kernel_size,kernel_stride,\
//                         stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
}


uint64_t CUTE_CONV_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    // printf("CUTE_CONV_MarcoTask\n");
    // printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    // printf("Conv_stride:%d,stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",conv_stride,stride_A,stride_B,stride_C,stride_D);
    // printf("conv_oh_index:%d,conv_ow_index:%d,conv_oh_max:%d,conv_ow_max:%d\n",conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max);



    //CUTE配置
    //CUTE指令发射

    /*
    注意实现，关于TensorLoad，Application_M可以不满64。
    Application_M不满64时，TileA的Load会提早结束。
    
        由于计算时，Application_M不满64，所以计算也会提早结束，但是计算部件是Matrix_M,Matrix_N,Matrix_K的计算部件，所以每次最少的计算单元是Matrix_M*Matrix_N*Matrix_K，
    此时，生成的结果为Matrix_M*Matrix_N*ResultWitdh的数据，会送入Reorder_FIFO,Reorder_FIFO正常会连续接受{CSP_DATAWIDTH/(Matrxi_N*ResultWidth)}个数据[目的是为了让一个CSP_DATAWIDTH的数据是Channel First的数据],
    然后完成重排序后送入VectorInterface，此时可加一个标志位确定是否需要后操作(填充的值直接不需要后操作,直接提早结束注意int32和int8需要补的拍),如果VectorInterface完成任务，则会输入到CSP_WRITE_FIFO中，CSP_WRITE_FIFO会连续接受数据，直到CSP_WRITE_FIFO填充至CSP_DATAWIDTH[VectroInterface的输出数据可能是量化后的数据所以bit数可能更低]，
    SCP_Write_FIFO根据送入的数据，计算偏移，让SCP内完全Matrix_N主序的数据(根据Matrix_N的大小，可能提前结束)。
    */

    // return issue_cute_conv_marco_inst(A,stride_A,B,stride_B,C,stride_C,D,stride_D,Application_M,Application_N,Application_K,kernel_stride,element_type,bias_type,transpose_result,conv_stride,conv_oh_max,conv_ow_max,kernel_size,CONV_OH_PER_ADD,CONV_OW_PER_ADD,conv_oh_index,conv_ow_index);

    // printf("invalid func[CUTE_CONV_MarcoTask]! wait YJP for finish it\n");
    exit(1);
    return -1;

}

uint64_t CUTE_CONV_1_1_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //任务和CUTE_MATMUL_MarcoTask是一样的
    // printf("CUTE_CONV_1_1_S1_MarcoTask\n");

    return CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,1,0,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

    
}

uint64_t CUTE_CONV_1_1_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //任务和CUTE_MATMUL_MarcoTask是一样的,只不过stride_A要乘2，且如果Application_M不是2的倍数，需要零填充，不能简单用CUTE_MATMUL_MarcoTask
    // printf("CUTE_CONV_1_1_S2_MarcoTask\n");

    // if(Application_M%2!=0 || Application_N%2!=0){
        // printf("[CUTE_CONV_1_1_S2_MarcoTask]Application_M/Application_N  is not a multiple of 2\n");
    //     exit(1);
    // }
    return CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,1,0,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
    
}

uint64_t CUTE_CONV_3_3_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //padding代表整个卷积任务是否存在padding，conv_oh_index,conv_ow_index代表当前卷积任务的第一个位置，我们的数据是output=[ohow][oc],input[ihiw][ic],kernel=[khkw][oc][ic]紧密排列的，所以可以判断是否需要进行0填充
    // printf("CUTE_CONV_3_3_S1_MarcoTask\n");

    //CUTE配置
    //CUTE指令发射

    /*
    kernel_size = 3的情况，就是9次kernel_size = 1的情况，但是由于只用存储一次C_SCP的数据，且有padding的填充任务，所以不能直接调用CUTE_CONV_1_1_S1_MarcoTask，微指令上有根本区别
    注意padding的信息，它描述的4个方向上是否有连续的padding，此时A的Load任务，需要处理好零填充的任务，直到0填充到Application_M的大小，然后再进行计算任务。
    */
    return CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,3,kernel_stride,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

}

uint64_t CUTE_CONV_3_3_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //padding代表整个卷积任务是否存在padding，我们的数据是output=[ohow][oc],input[ihiw][ic],kernel=[khkw][oc][ic]紧密排列的，所以可以判断是否需要进行0填充
    // printf("CUTE_CONV_3_3_S2_MarcoTask\n");

    //CUTE配置
    //CUTE指令发射

    /*
    kernel_size = 3的情况已经在CUTE_CONV_3_3_S1_MarcoTask中讨论过了，现在stride=2，
    需要计算ow,oh->根据stride计算当前中心点->当前处理的conv_1_1任务位点(9次中的哪一次),计算当前的conv_1_1在A中Load的任务位点，然后决定是否需要padding处理
    */
    return CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,3,kernel_stride,\
                        stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
}

void CUTE_TASK_END(uint64_t task_id)
{
    // // printf("waiting for task end\n");
    // //等待任务结束
    // uint64_t finish_tag = 1 << task_id;
    // uint64_t res1 = cute_marco_inst_fifo_finish_search();
    // while(!(res1&finish_tag))
    // {
    //     // printf("Waiting for finish task_id = %d\n",task_id);
    //     res1 = cute_marco_inst_fifo_finish_search();
    // }
    // cute_marco_inst_fifo_dequeue();
    // // printf("Task End\n");
    return;
}

void CUTE_CONV_3_3_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    // printf("CUTE_CONV_3_3_S2_AUTO\n");
    fflush(stdout);

    int batches = params.batch_size;
    int CONV_Matrix_M = params.out_row_dim * params.out_col_dim;
    int CONV_Matrix_N = params.out_channels;
    int CONV_Matrix_K = params.in_channels;

    int CONV_Current_Matrix_M = 0;
    int CONV_Current_Matrix_N = 0;
    int CONV_Current_Matrix_K = 0;

    int CONV_Current_oh_index = 0;
    int CONV_Current_ow_index = 0;
    int CONV_Current_oh_max = params.out_row_dim;
    int CONV_Current_ow_max = params.out_col_dim;

    uint64_t input_batch_stride = params.in_channels * params.in_col_dim * params.in_row_dim;
    uint64_t output_batch_stride = params.out_channels * params.out_col_dim * params.out_row_dim;

    //afater_operation
    void (*afater_u_kernel_operation)(int32_t * , int8_t * , uint64_t , uint64_t ,uint64_t , uint64_t ) = act_type==RELU?scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x:scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_relu_shift_x;
    void (*afater_operation)(acc_t *,int,int,elem_t *,int,int) = act_type==RELU?scale_after_operation_64_64_relu:scale_after_operation_64_64;
    //遍历batch
    for(int i=0;i<batches;i++)
    {
        //input = [ih,iw][ic]
        //weights = [kh,kw][oc][ic]
        //output = [oh,ow][oc]

        int CONV_Current_Tile_M = 0;//输出矩阵分块的M坐标
        int CONV_Current_Tile_N = 0;//输出矩阵分块的N坐标
        // int CONV_Current_Tile_K = 0;//输出矩阵分块的K坐标
        int CONV_Current_Tile_M_Max = CONV_Matrix_M/CUTE_TILE_Tensor_M + (CONV_Matrix_M%CUTE_TILE_Tensor_M!=0);
        int CONV_Current_Tile_N_Max = CONV_Matrix_N/CUTE_TILE_Tensor_N + (CONV_Matrix_N%CUTE_TILE_Tensor_N!=0);//resnet50不会发生
        // int CONV_Current_Tile_K_Max = CONV_Matrix_K/CUTE_TILE_Tensor_K + (CONV_Matrix_K%CUTE_TILE_Tensor_K!=0);//resnet50不会发生

        bool Has_Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M!=0;
        // bool Has_Last_Tile_N = CONV_Matrix_N%CUTE_TILE_Tensor_N!=0;
        // bool Has_Last_Tile_K = CONV_Matrix_K%CUTE_TILE_Tensor_K!=0;
        int Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M;
        // int Last_Tile_N = CONV_Current_Matrix_N%CONV_Matrix_N;
        // int Last_Tile_K = CONV_Current_Matrix_K%CONV_Matrix_K;

        bool have_after_operation = false;
        uint64_t wait_after_operation_cute_task_id = 0;

        elem_t *VECTASK_C_Addr = output;
        uint64_t VECTASK_C_stride = params.out_channels;
        uint64_t VECTASK_CUTE_result_stride = params.out_channels*4;//int32->int8
        int VECTASK_DIM_I = 0;
        int VECTASK_DIM_J = 0;
        int VECTASK_RESULT_INDEX = 0;

        //遍历矩阵分块
        for(CONV_Current_Tile_M=0;CONV_Current_Tile_M<CONV_Current_Tile_M_Max;CONV_Current_Tile_M++)
        {
            for(CONV_Current_Tile_N=0;CONV_Current_Tile_N<CONV_Current_Tile_N_Max;CONV_Current_Tile_N++)
            {
                //计算当前矩阵分块的M对应的oh,ow
                int CONV_Current_oh = (CONV_Current_Tile_M*64)/params.out_col_dim;
                int CONV_Current_ow = (CONV_Current_Tile_M*64)%params.out_col_dim;
                //计算当前矩阵分块的N对应的oc
                int CONV_Current_oc = CONV_Current_Tile_N*64;

                //(kernel,stride) = （1,1)很简单
                int CONV_Current_ih = CONV_Current_oh;
                int CONV_Current_iw = CONV_Current_ow;

                bool Is_Last_Tile_M = Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1;

                int Application_M = (Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1)?Last_Tile_M:CUTE_TILE_Tensor_M;
                int Application_N = CUTE_TILE_Tensor_N;
                int Application_K = CONV_Matrix_K;
                void *A = input + input_batch_stride*i;
                void *B = weights + CONV_Current_Tile_N * CUTE_TILE_Tensor_N * params.in_channels;
                void *C = CUTE_result[CUTE_result_index];
                void *D = bias + CONV_Current_Tile_N * CUTE_TILE_Tensor_N;
                int element_type = CUTE_INT8;
                int bias_type = BIAS_TYPE;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                uint64_t kernel_stride = params.in_channels*params.out_channels;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                wait_after_operation_cute_task_id = CUTE_CONV_3_3_S2_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,2,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END(wait_after_operation_cute_task_id);
                    // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
                }
                have_after_operation = true;
                VECTASK_RESULT_INDEX = CUTE_result_index;
                CUTE_result_index = CUTE_result_index^1;
                VECTASK_C_Addr = output + CUTE_TILE_Tensor_M*CONV_Current_Tile_M*VECTASK_C_stride+CONV_Current_Tile_N*CUTE_TILE_Tensor_N + output_batch_stride*i;
                VECTASK_DIM_I = Application_M;
                VECTASK_DIM_J = Application_N;
            }
        }
        if(have_after_operation)
        {
            // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
            afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
        }
    }
}
void CUTE_CONV_3_3_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    // printf("CUTE_CONV_3_3_S1_AUTO\n");
    fflush(stdout);

    int batches = params.batch_size;
    int CONV_Matrix_M = params.out_row_dim * params.out_col_dim;
    int CONV_Matrix_N = params.out_channels;
    int CONV_Matrix_K = params.in_channels;

    int CONV_Current_Matrix_M = 0;
    int CONV_Current_Matrix_N = 0;
    int CONV_Current_Matrix_K = 0;

    int CONV_Current_oh_index = 0;
    int CONV_Current_ow_index = 0;
    int CONV_Current_oh_max = params.out_row_dim;
    int CONV_Current_ow_max = params.out_col_dim;

    uint64_t input_batch_stride = params.in_channels * params.in_col_dim * params.in_row_dim;
    uint64_t output_batch_stride = params.out_channels * params.out_col_dim * params.out_row_dim;

    //afater_operation
    void (*afater_u_kernel_operation)(int32_t * , int8_t * , uint64_t , uint64_t ,uint64_t , uint64_t ) = act_type==RELU?scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x:scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_relu_shift_x;
    void (*afater_operation)(acc_t *,int,int,elem_t *,int,int) = act_type==RELU?scale_after_operation_64_64_relu:scale_after_operation_64_64;
    //遍历batch
    for(int i=0;i<batches;i++)
    {
        //input = [ih,iw][ic]
        //weights = [kh,kw][oc][ic]
        //output = [oh,ow][oc]

        int CONV_Current_Tile_M = 0;//输出矩阵分块的M坐标
        int CONV_Current_Tile_N = 0;//输出矩阵分块的N坐标
        // int CONV_Current_Tile_K = 0;//输出矩阵分块的K坐标
        int CONV_Current_Tile_M_Max = CONV_Matrix_M/CUTE_TILE_Tensor_M + (CONV_Matrix_M%CUTE_TILE_Tensor_M!=0);
        int CONV_Current_Tile_N_Max = CONV_Matrix_N/CUTE_TILE_Tensor_N + (CONV_Matrix_N%CUTE_TILE_Tensor_N!=0);//resnet50不会发生
        // int CONV_Current_Tile_K_Max = CONV_Matrix_K/CUTE_TILE_Tensor_K + (CONV_Matrix_K%CUTE_TILE_Tensor_K!=0);//resnet50不会发生

        bool Has_Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M!=0;
        // bool Has_Last_Tile_N = CONV_Matrix_N%CUTE_TILE_Tensor_N!=0;
        // bool Has_Last_Tile_K = CONV_Matrix_K%CUTE_TILE_Tensor_K!=0;
        int Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M;
        // int Last_Tile_N = CONV_Current_Matrix_N%CONV_Matrix_N;
        // int Last_Tile_K = CONV_Current_Matrix_K%CONV_Matrix_K;

        bool have_after_operation = false;
        uint64_t wait_after_operation_cute_task_id = 0;
        elem_t *VECTASK_C_Addr = output;
        uint64_t VECTASK_C_stride = params.out_channels;
        uint64_t VECTASK_CUTE_result_stride = params.out_channels*4;//int32->int8
        int VECTASK_DIM_I = 64;
        int VECTASK_DIM_J = 64;
        int VECTASK_RESULT_INDEX = 0;

        //遍历矩阵分块
        for(CONV_Current_Tile_M=0;CONV_Current_Tile_M<CONV_Current_Tile_M_Max;CONV_Current_Tile_M++)
        {
            for(CONV_Current_Tile_N=0;CONV_Current_Tile_N<CONV_Current_Tile_N_Max;CONV_Current_Tile_N++)
            {
                //计算当前矩阵分块的M对应的oh,ow
                int CONV_Current_oh = (CONV_Current_Tile_M*64)/params.out_col_dim;
                int CONV_Current_ow = (CONV_Current_Tile_M*64)%params.out_col_dim;
                //计算当前矩阵分块的N对应的oc
                int CONV_Current_oc = CONV_Current_Tile_N*64;

                //(kernel,stride) = （1,1)很简单
                int CONV_Current_ih = CONV_Current_oh;
                int CONV_Current_iw = CONV_Current_ow;

                bool Is_Last_Tile_M = Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1;

                int Application_M = (Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1)?Last_Tile_M:CUTE_TILE_Tensor_M;
                int Application_N = CUTE_TILE_Tensor_N;
                int Application_K = CONV_Matrix_K;
                void *A = input + input_batch_stride*i;
                void *B = weights + CONV_Current_Tile_N * CUTE_TILE_Tensor_N * params.in_channels;
                void *C = CUTE_result[CUTE_result_index];
                void *D = bias + CONV_Current_Tile_N * CUTE_TILE_Tensor_N;
                int element_type = CUTE_INT8;
                int bias_type = BIAS_TYPE;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                uint64_t kernel_stride = params.in_channels*params.out_channels;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                wait_after_operation_cute_task_id = CUTE_CONV_3_3_S1_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                // printf("wait_after_operation_cute_task_id = %d\n",wait_after_operation_cute_task_id);
                if(have_after_operation)
                {
                    CUTE_TASK_END(wait_after_operation_cute_task_id);
                    // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
                }
                have_after_operation = true;
                VECTASK_RESULT_INDEX = CUTE_result_index;
                CUTE_result_index = CUTE_result_index^1;
                VECTASK_C_Addr = output + CUTE_TILE_Tensor_M*CONV_Current_Tile_M*VECTASK_C_stride+CONV_Current_Tile_N*CUTE_TILE_Tensor_N + output_batch_stride*i;
                VECTASK_DIM_I = Application_M;
                VECTASK_DIM_J = Application_N;
            }
        }
        if(have_after_operation)
        {
            // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
        }
    }
}
void CUTE_CONV_1_1_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    // printf("CUTE_CONV_1_1_S2_AUTO\n");
    fflush(stdout);
    int batches = params.batch_size;
    int CONV_Matrix_M = params.out_row_dim * params.out_col_dim;
    int CONV_Matrix_N = params.out_channels;
    int CONV_Matrix_K = params.in_channels;

    int CONV_Current_Matrix_M = 0;
    int CONV_Current_Matrix_N = 0;
    int CONV_Current_Matrix_K = 0;

    uint64_t input_batch_stride = params.in_channels * params.in_col_dim * params.in_row_dim;
    uint64_t output_batch_stride = params.out_channels * params.out_col_dim * params.out_row_dim;

    //afater_operation
    void (*afater_u_kernel_operation)(int32_t * , int8_t * , uint64_t , uint64_t ,uint64_t , uint64_t ) = act_type==RELU?scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x:scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_relu_shift_x;
    void (*afater_operation)(acc_t *,int,int,elem_t *,int,int) = act_type==RELU?scale_after_operation_64_64_relu:scale_after_operation_64_64;
    //遍历batch
    for(int i=0;i<batches;i++)
    {
        //input = [ih,iw][ic]
        //weights = [kh,kw][oc][ic]
        //output = [oh,ow][oc]

        int CONV_Current_Tile_M = 0;//输出矩阵分块的M坐标
        int CONV_Current_Tile_N = 0;//输出矩阵分块的N坐标
        // int CONV_Current_Tile_K = 0;//输出矩阵分块的K坐标
        int CONV_Current_Tile_M_Max = CONV_Matrix_M/CUTE_TILE_Tensor_M + (CONV_Matrix_M%CUTE_TILE_Tensor_M!=0);
        int CONV_Current_Tile_N_Max = CONV_Matrix_N/CUTE_TILE_Tensor_N + (CONV_Matrix_N%CUTE_TILE_Tensor_N!=0);//resnet50不会发生
        // int CONV_Current_Tile_K_Max = CONV_Matrix_K/CUTE_TILE_Tensor_K + (CONV_Matrix_K%CUTE_TILE_Tensor_K!=0);//resnet50不会发生

        bool Has_Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M!=0;
        // bool Has_Last_Tile_N = CONV_Matrix_N%CUTE_TILE_Tensor_N!=0;
        // bool Has_Last_Tile_K = CONV_Matrix_K%CUTE_TILE_Tensor_K!=0;
        int Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M;
        // int Last_Tile_N = CONV_Current_Matrix_N%CONV_Matrix_N;
        // int Last_Tile_K = CONV_Current_Matrix_K%CONV_Matrix_K;

        bool have_after_operation = false;
        uint64_t wait_after_operation_cute_task_id = 0;
        elem_t *VECTASK_C_Addr = output;
        uint64_t VECTASK_C_stride = params.out_channels;
        uint64_t VECTASK_CUTE_result_stride = params.out_channels*4;//int32->int8
        int VECTASK_DIM_I = 64;
        int VECTASK_DIM_J = 64;
        int VECTASK_RESULT_INDEX = 0;

        //遍历矩阵分块
        for(CONV_Current_Tile_M=0;CONV_Current_Tile_M<CONV_Current_Tile_M_Max;CONV_Current_Tile_M++)
        {
            for(CONV_Current_Tile_N=0;CONV_Current_Tile_N<CONV_Current_Tile_N_Max;CONV_Current_Tile_N++)
            {
                //计算当前矩阵分块的M对应的oh,ow
                int CONV_Current_oh = (CONV_Current_Tile_M*64)/params.out_col_dim;
                int CONV_Current_ow = (CONV_Current_Tile_M*64)%params.out_col_dim;
                //计算当前矩阵分块的N对应的oc
                int CONV_Current_oc = CONV_Current_Tile_N*64;

                int current_ih = CONV_Current_oh*2;
                int current_iw = CONV_Current_ow*2;
                // printf("[TILE_S2]current_ih:%d,current_iw:%d\n",current_ih,current_iw);


                bool Is_Last_Tile_M = Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1;

                int Application_M = (Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1)?Last_Tile_M:CUTE_TILE_Tensor_M;
                int Application_N = CUTE_TILE_Tensor_N;
                int Application_K = CONV_Matrix_K;
                void *A = input + i*input_batch_stride;//计算的时候会利用oh和ow，直接传input就好
                void *B = weights + CONV_Current_Tile_N * CUTE_TILE_Tensor_N * params.in_channels;
                void *C = CUTE_result[CUTE_result_index];
                void *D = bias + CONV_Current_Tile_N * CUTE_TILE_Tensor_N;
                int element_type = CUTE_INT8;
                int bias_type = BIAS_TYPE;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                wait_after_operation_cute_task_id = CUTE_CONV_1_1_S2_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,2,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END(wait_after_operation_cute_task_id);
                    // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
                }
                have_after_operation = true;
                VECTASK_RESULT_INDEX = CUTE_result_index;
                CUTE_result_index = CUTE_result_index^1;
                VECTASK_C_Addr = output + CUTE_TILE_Tensor_M*CONV_Current_Tile_M*VECTASK_C_stride+CONV_Current_Tile_N*CUTE_TILE_Tensor_N + output_batch_stride*i;
                VECTASK_DIM_I = Application_M;
                VECTASK_DIM_J = Application_N;
            }
        }
        if(have_after_operation)
        {
            // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
        }
    }
}

void CUTE_CONV_1_1_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    // printf("CUTE_CONV_1_1_S1_AUTO\n");
    // fflush(stdout);
    int batches = params.batch_size;
    int CONV_Matrix_M = params.out_row_dim * params.out_col_dim;
    int CONV_Matrix_N = params.out_channels;
    int CONV_Matrix_K = params.in_channels;

    int CONV_Current_Matrix_M = 0;
    int CONV_Current_Matrix_N = 0;
    int CONV_Current_Matrix_K = 0;

    int CONV_Current_oh_index = 0;
    int CONV_Current_ow_index = 0;
    int CONV_Current_oh_max = params.out_row_dim;
    int CONV_Current_ow_max = params.out_col_dim;

    uint64_t input_batch_stride = params.in_channels * params.in_col_dim * params.in_row_dim;
    uint64_t output_batch_stride = params.out_channels * params.out_col_dim * params.out_row_dim;

    //afater_operation
    void (*afater_u_kernel_operation)(int32_t * , int8_t * , uint64_t , uint64_t ,uint64_t , uint64_t ) = act_type==RELU?scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_shift_x:scale_after_operation_i32_to_i8_DimI_x_DimJ_64_ukernel_relu_shift_x;
    void (*afater_operation)(acc_t *,int,int,elem_t *,int,int) = act_type==RELU?scale_after_operation_64_64_relu:scale_after_operation_64_64;
    //遍历batch
    // batches = 1;//TODO:
    for(int i=0;i<batches;i++)
    {
        //input = [ih,iw][ic]
        //weights = [kh,kw][oc][ic]
        //output = [oh,ow][oc]

        int CONV_Current_Tile_M = 0;//输出矩阵分块的M坐标
        int CONV_Current_Tile_N = 0;//输出矩阵分块的N坐标
        // int CONV_Current_Tile_K = 0;//输出矩阵分块的K坐标
        int CONV_Current_Tile_M_Max = CONV_Matrix_M/CUTE_TILE_Tensor_M + (CONV_Matrix_M%CUTE_TILE_Tensor_M!=0);
        int CONV_Current_Tile_N_Max = CONV_Matrix_N/CUTE_TILE_Tensor_N + (CONV_Matrix_N%CUTE_TILE_Tensor_N!=0);//resnet50不会发生
        // int CONV_Current_Tile_K_Max = CONV_Matrix_K/CUTE_TILE_Tensor_K + (CONV_Matrix_K%CUTE_TILE_Tensor_K!=0);//resnet50不会发生

        bool Has_Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M!=0;
        // bool Has_Last_Tile_N = CONV_Matrix_N%CUTE_TILE_Tensor_N!=0;
        // bool Has_Last_Tile_K = CONV_Matrix_K%CUTE_TILE_Tensor_K!=0;
        int Last_Tile_M = CONV_Matrix_M%CUTE_TILE_Tensor_M;
        // int Last_Tile_N = CONV_Current_Matrix_N%CONV_Matrix_N;
        // int Last_Tile_K = CONV_Current_Matrix_K%CONV_Matrix_K;

        bool have_after_operation = false;
        uint64_t wait_after_operation_cute_task_id = 0;

        elem_t *VECTASK_C_Addr = output;
        uint64_t VECTASK_C_stride = params.out_channels;
        uint64_t VECTASK_CUTE_result_stride = params.out_channels*4;//int32->int8
        int VECTASK_DIM_I = 64;
        int VECTASK_DIM_J = 64;
        int VECTASK_RESULT_INDEX = 0;

        //遍历矩阵分块
        for(CONV_Current_Tile_M=0;CONV_Current_Tile_M<CONV_Current_Tile_M_Max;CONV_Current_Tile_M++)
        {
            for(CONV_Current_Tile_N=0;CONV_Current_Tile_N<CONV_Current_Tile_N_Max;CONV_Current_Tile_N++)
            {
                //计算当前矩阵分块的M对应的oh,ow
                int CONV_Current_oh = (CONV_Current_Tile_M*64)/params.out_col_dim;
                int CONV_Current_ow = (CONV_Current_Tile_M*64)%params.out_col_dim;
                //计算当前矩阵分块的N对应的oc
                int CONV_Current_oc = CONV_Current_Tile_N*64;

                //(kernel,stride) = （1,1)很简单
                int CONV_Current_ih = CONV_Current_oh;
                int CONV_Current_iw = CONV_Current_ow;

                bool Is_Last_Tile_M = Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1;

                int Application_M = (Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1)?Last_Tile_M:CUTE_TILE_Tensor_M;
                int Application_N = CUTE_TILE_Tensor_N;
                int Application_K = CONV_Matrix_K;
                void *A = input + input_batch_stride*i;
                void *B = weights + CONV_Current_Tile_N * CUTE_TILE_Tensor_N * params.in_channels;
                void *C = CUTE_result[CUTE_result_index];
                void *D = bias + CONV_Current_Tile_N * CUTE_TILE_Tensor_N;
                int element_type = CUTE_INT8;
                int bias_type = BIAS_TYPE;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                wait_after_operation_cute_task_id = CUTE_CONV_1_1_S1_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END(wait_after_operation_cute_task_id);
                    // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
                }
                have_after_operation = true;
                VECTASK_RESULT_INDEX = CUTE_result_index;
                CUTE_result_index = CUTE_result_index^1;
                VECTASK_C_Addr = output + CUTE_TILE_Tensor_M*CONV_Current_Tile_M*VECTASK_C_stride+CONV_Current_Tile_N*CUTE_TILE_Tensor_N + output_batch_stride*i;
                VECTASK_DIM_I = Application_M;
                VECTASK_DIM_J = Application_N;
            }
        }
        if(have_after_operation)
        {
            // afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
                    afater_u_kernel_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_C_Addr,64,CONV_Matrix_N,params.output_scale_shift,VECTASK_DIM_I);
        }
    }
}

static void tiled_conv_CUTE_auto(ConvParams params,
        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,
        int act_type)
{
    // printf("CUTE_CONV_AUTO\n");
    // fflush(stdout);
    if(params.kernel_size == 1 && params.stride == 1)
    {
        CUTE_CONV_1_1_S1_AUTO(params, input, weights, bias, output,act_type);
    }
    else if(params.kernel_size == 1 && params.stride == 2)
    {
        CUTE_CONV_1_1_S2_AUTO(params, input, weights, bias, output,act_type);
    }
    else if(params.kernel_size == 3 && params.stride == 1)
    {
        CUTE_CONV_3_3_S1_AUTO(params, input, weights, bias, output,act_type);
    }
    else if(params.kernel_size == 3 && params.stride == 2)
    {
        CUTE_CONV_3_3_S2_AUTO(params, input, weights, bias, output,act_type);
    }
    else
    {
        printf("CUTE_CONV_AUTO not implemented for this configuration\n");
        exit(1);
    }

}

int main (int argc, char * argv[]) {

    /*Hello world from core 0???*/
  uint64_t marchid = read_csr(marchid);
  const char* march = get_march(marchid);
  printf("Hello world from core 0, a %s\n", march);
  //输出mstatus,16进制
    unsigned long mstatus;
    asm volatile ("csrr %0, mstatus" : "=r" (mstatus));
    printf("%lx\n", mstatus);
 //设置mstatus.VS = 1，其中mstatus[10:9]为mstatus.VS
    asm volatile ("csrw mstatus, %0" : : "r" (mstatus | (1 << 9)));
    asm volatile ("csrr %0, mstatus" : "=r" (mstatus));
    printf("%lx\n", mstatus);

    // conv_2
    uint64_t start = read_cycles();
    tiled_conv_CUTE_auto(conv_48_params, input, weights, bias, output,ACT_TYPE);
    uint64_t end = read_cycles();
    printf("conv_48 no cute cycles: %lu \n", end - start);
    printf("conv_48 vec cycles: %lu \n",vec_time);

    printf("PASS\n");
    exit(0);
}

