#include <stdio.h>
#include <string.h>
#include <stdbool.h>
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

// #include <stdint.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>

// #define XCUSTOM_ACC 3
// #define DIM 16
// #define ADDR_LEN 32
// #define BANK_NUM 4
// #define BANK_ROWS 4096
// #define ACC_ROWS 1024
// #define MAX_BYTES 64
// #define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
// #define MAX_BLOCK_LEN_ACC (MAX_BYTES/(DIM*4))

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

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#define MVIN_SCALE_IDENTITY 1.0

#define ACC_SCALE_IDENTITY 1.0

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT(x, shift) \
    ((shift) > 0 ? (((x) >> (shift)) + \
        (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
             ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#ifdef __cplusplus
#define SAME_TYPE(x) decltype(x)
#else
#define SAME_TYPE(x) typeof(x)
#endif

#define ROUND_NEAR_EVEN(x) \
    ({ const SAME_TYPE(x) x_ = (x); \
         const long long i = x_; \
         const long long next = x_ < 0 ? x_ - 1 : x_ + 1; \
         SAME_TYPE(x) rem = x_ - i; \
         rem = rem < 0 ? -rem : rem; \
         SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( \
                     i % 2 == 0 ? i : next)); \
         result; })

// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm
#define ROUNDING_RIGHT_SHIFT_BITS(x, shift) \
((shift) > 0 ? (((x) >> (shift)) + \
    (((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \
         ((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift))))

#define ACC_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);})

#define MVIN_SCALE(x, scale) \
    ({float y = ROUND_NEAR_EVEN((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (elem_t)y);})

#define MVIN_SCALE_ACC(x, scale) (x)

#define ACC_SCALE_T_IS_FLOAT
#define ACC_SCALE_EXP_BITS 8
#define ACC_SCALE_SIG_BITS 24

#define ACC_READ_SMALL_WIDTH
#define ACC_READ_FULL_WIDTH

#define HAS_FIRST_LAYER_OPTIMIZATIONS

#endif // GEMMINI_PARAMS_H

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#include <math.h>

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

#define gemmini_fence() 

#define GEMMINI_ACC_SCALE(x, scale) (x)

#define GEMMINI_SCALE(x, scale) (x)


static acc_t int_sqrt(acc_t n) {
  if (n == 0) return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1) {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev) return x_prev;
    x_prev = x_next;
  };
}


static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale) {
  // Apply I-GELU if needed
  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU) {
    x = x < 0 ? 0 : x;
  }
  return x;
}


enum tiled_matmul_type_t {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
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

static void tiled_resadd_auto(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

	  resadd_cpu(I, J, A_scale, B_scale, C_scale,
		A, B, C, relu);
}
// 512位对齐的数组acc_t result[64][64]
acc_t __attribute__((aligned(512))) CUTE_result[2][64][3072];//double buffer
int CUTE_result_index = 0;

void softmax_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int) (0.693147 / scale)==0?1:(int) (0.693147 / scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / scale;
    const acc_t qc = c / (a*scale*scale);


    //normal softmax
    for (size_t i = 0; i < dim_i; i++) {

        //round 1
        acc_t max_q = -2147483648;
        for (size_t j = 0; j < dim_j; j++) 
        {
            if (input[i][j] > max_q) max_q = input[i][j];
        }
        //round 2
        acc_t sum_exp = 0;
        for (size_t j = 0; j < dim_j; j++) 
        {
            acc_t q = input[i][j] - max_q;
            acc_t z = (acc_t) (-q * qln2_inv) >> 16;
            acc_t qp = q + z * qln2;
            acc_t q_exp = (qp + qb)*(qp + qb) + qc;
            input[i][j] = q_exp >> z;
            sum_exp += input[i][j];
        }
            

        //round 3
        scale_t factor = (127.f) / (float) sum_exp;
        for (size_t j = 0; j < dim_j; j++) 
        {
            elem_t* c = output + i * stride_c + j;
            acc_t x = input[i][j];
            x = x * factor;
            // Clip result
            x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
            *c = x;
        }
    }

    //vector softmax
    for (size_t i = 0; i < dim_i; i++) {
        
        //round 1
        acc_t max_q[8] = {-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648,-2147483648};
        for (size_t j = 0; j < dim_j/8; j++) 
        {
            for (size_t k = 0; k < 8; k++) 
            {
                if (input[i][j*8+k] > max_q[k]) max_q[k] = input[i][j*8+k];
            }
        }
        acc_t max_q_max = -2147483648;
        for (size_t k = 0; k < 8; k++)
        {
            if (max_q[k] > max_q_max) max_q_max = max_q[k];
        }

        //round 2
        acc_t sum_exp[8] = {0,0,0,0,0,0,0,0};
        for (size_t j = 0; j < dim_j/8; j++) 
        {
            for (size_t k = 0; k < 8; k++)
            {
                acc_t q = input[i][j*8+k] - max_q_max;
                acc_t z = (acc_t) (-q * qln2_inv) >> 16;
                acc_t qp = q + z * qln2;
                acc_t q_exp = (qp + qb)*(qp + qb) + qc;
                input[i][j*8+k] = q_exp >> z;
                sum_exp[k] += input[i][j*8+k];
            }
        }
        acc_t sum_exp_sum = 0;
        for (size_t k = 0; k < 8; k++)
        {
            sum_exp_sum += sum_exp[k];
        }

        scale_t factor = (127.f) / (float) sum_exp_sum;
        //round 3
        scale_t factor_8[8] = {factor,factor,factor,factor,factor,factor,factor,factor};
        for (size_t j = 0; j < dim_j/8; j++) 
        {
            for (size_t k = 0; k < 8; k++)
            {
                elem_t* c = output + i * stride_c + j*8+k;
                acc_t x = input[i][j*8+k];
                x = x * factor_8[k];
                // Clip result
                x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
                *c = x;
            }
        }
    }

}

void layernorm_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);
    
    //normal layer norm
    for(size_t i = 0; i < dim_i; i++)
    {
        acc_t sum = 0;
        for (size_t j = 0; j < dim_j; j++)
            sum += input[i][j];
        acc_t mean = sum / (acc_t)dim_j;

        acc_t total_err_sq = 0;
        for (size_t j = 0; j < dim_j; j++)
            total_err_sq += (input[i][j] - mean)*(input[i][j] - mean);
        acc_t variance = total_err_sq / (acc_t)dim_j;

        acc_t stddev = int_sqrt(variance); //最好有专用指令，不然没法向量化
        if (variance == 0) stddev = 1;


        for (size_t j = 0; j < dim_j; j++) {

            elem_t* c = output + i * stride_c + j;
            acc_t x = input[i][j];
            x -= mean;
            x /= stddev*scale;
            // Clip result
            x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
            *c = x;
        }
    }

    //vector layer norm
    for (size_t i = 0; i < dim_i; i++) {
        
        acc_t sum[8] = {0,0,0,0,0,0,0,0};
        for (size_t j = 0; j < dim_j/8; j++)
        {
            for (size_t k = 0; k < 8; k++)
            {
                sum[k] += input[i][j*8+k];
            }
        }

        acc_t sum_sum = 0;
        for (size_t k = 0; k < 8; k++)
        {
            sum_sum += sum[k];
        }

        acc_t t_mean = sum_sum / (acc_t)dim_j;
        acc_t mean[8] = {t_mean,t_mean,t_mean,t_mean,t_mean,t_mean,t_mean,t_mean};

        acc_t total_err_sq[8] = {0,0,0,0,0,0,0,0};
        for (size_t j = 0; j < dim_j/8; j++)
        {
            for (size_t k = 0; k < 8; k++)
            {
                total_err_sq[k] += (input[i][j*8+k] - mean[k])*(input[i][j*8+k] - mean[k]);
            }
        }
        acc_t total_err_sq_sum = 0;
        for (size_t k = 0; k < 8; k++)
        {
            total_err_sq_sum += total_err_sq[k];
        }

        acc_t variance = total_err_sq_sum / (acc_t)dim_j;

        acc_t stddev_t = int_sqrt(variance);
        if (variance == 0) stddev_t = 1;

        acc_t stddev[8] = {stddev_t,stddev_t,stddev_t,stddev_t,stddev_t,stddev_t,stddev_t,stddev_t};

        for (size_t j = 0; j < dim_j/8; j++)
        {
            for (size_t k = 0; k < 8; k++)
            {
                elem_t* c = output + i * stride_c + j*8+k;
                acc_t x = input[i][j*8+k];
                x -= mean[k];
                x /= stddev[k]*scale;
                // Clip result
                x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
                *c = x;
            }
        }
    }

}

void Gelu_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);

    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = scale;

    const acc_scale_t S_erf = (-0.2888 * (S/sqrt_2)*(S/sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    //normal I-Gelu
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j++) {
            acc_t q = input[i][j];
            acc_t q_sign = q < 0 ? -1 : 1;
            acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
            acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
            acc_t q_erf = q_sign * q_poly;
            q = q * (q_erf + q1);
            elem_t* c = output + i * stride_c + j;
            // Clip result
            q = q > elem_t_max ? elem_t_max : (q < elem_t_min ? elem_t_min : q);
            *c = q;
        }
    }

    //Vector I-Gelu
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j+=8) {
            for (size_t k = 0; k < 8; k++) {
                acc_t q = input[i][j+k];
                acc_t q_sign = q < 0 ? -1 : 1;
                acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
                acc_t q_poly = (q_clipped + qb)*(q_clipped + qb) + qc;
                acc_t q_erf = q_sign * q_poly;
                q = q * (q_erf + q1);
                elem_t* c = output + i * stride_c + j+k;
                // Clip result
                q = q > elem_t_max ? elem_t_max : (q < elem_t_min ? elem_t_min : q);
                *c = q;
            }
        }
    }



    //pi = 3.14159265359;
    //Gelu(x) = 0.5*x*(1 + tanh(-sqrt(2/pi) * (x + 0.044715 * x^3)))
    //normal Gelu
    const acc_scale_t sqrt_2_div_pi = 0.797884561;
    const acc_scale_t a = 0.044715;
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j++) {
            acc_t q = input[i][j];
            acc_t q_3 = (acc_t) (q * q * q);
            acc_scale_t q_tanh_input = sqrt_2_div_pi * (q + a * q_3);
            acc_t q_tanh = tanh(q_tanh_input);//最好有专用指令
            q = q * (0.5 * (1 + q_tanh));
            elem_t* c = output + i * stride_c + j;
            // Clip result
            q = q > elem_t_max ? elem_t_max : (q < elem_t_min ? elem_t_min : q);
            *c = q;
        }
    }

    //Vector Gelu
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j+=8) {
            for (size_t k = 0; k < 8; k++) {
                acc_t q = input[i][j+k];
                acc_t q_3 = (acc_t) (q * q * q);
                acc_scale_t q_tanh_input = sqrt_2_div_pi * (q + a * q_3);
                acc_t q_tanh = tanh(q_tanh_input);//最好有专用指令
                q = q * (0.5 * (1 + q_tanh));
                elem_t* c = output + i * stride_c + j+k;
                // Clip result
                q = q > elem_t_max ? elem_t_max : (q < elem_t_min ? elem_t_min : q);
                *c = q;
            }
        }
    }


}

void scale_after_operation(acc_t  input[64][3072], int dim_i,int dim_j,elem_t * output,acc_scale_t scale,int stride_c)
{

    //输出所有函数输入
    printf("dim_i:%d,dim_j:%d,scale:%f,stride_c:%d\n",dim_i,dim_j,scale,stride_c);

    //normal scale
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j++) {
            elem_t* c = output + i * stride_c + j;
            acc_t x = input[i][j];
            x = x * scale;
            // Clip result
            x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
            *c = x;
        }
    }

    //vector scale
    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j+=8) {
            for (size_t k = 0; k < 8; k++) {
                elem_t* c = output + i * stride_c + j+k;
                acc_t x = input[i][j+k];
                x = x * scale;
                // Clip result
                x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
                *c = x;
            }
        }
    }

}


void MATMUL_MARCO_ISSUE()
{
    printf("MATMUL_MARCO_ISSUE\n");
    // exit(1);
}

int MATMUL_MARCO_SEARCH()
{
    printf("MATMUL_MARCO_SEARCH\n");
    return (rand()%100)<17;
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
    printf("transA:%d,transB:%d,DIM_I:%d,DIM_J:%d,DIM_K:%d,stride_A:%d,stride_B:%d,stride_D:%d,stride_C:%d\nA_scale_factor:%f,B_scale_factor:%f,D_scale_factor:%d,act:%d,scale:%f,bert_scale:%f,repeating_bias:%d\n",transA,transB,DIM_I,DIM_J,DIM_K,stride_A,stride_B,stride_D,stride_C,A_scale_factor,B_scale_factor,D_scale_factor,act,scale,bert_scale,repeating_bias);
  //如果不是layernorm或者softmax切成64,64,K的小块，然后每次完成计算，调用向量算子。
  //如果是layernorm或者softmax，或者不是64,M,K的小块，然后调用向量算子。

  if(!(DIM_I % 64 == 0 && DIM_J % 64 == 0 && DIM_K % 64 == 0))
  {
    printf("Can't Till Now!");
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
      afater_operation = scale_after_operation;
      break;
    case RELU:
      afater_operation = NULL;
      break;
    case LAYERNORM:
      afater_operation = layernorm_after_operation;
      break;
    case IGELU:
      afater_operation = Gelu_after_operation;
      break;
    case SOFTMAX:
      afater_operation = softmax_after_operation;
      break;
    default:
      afater_operation = scale_after_operation;
      break;
  }
  
  printf("!!\n[matmul_cute] START!!\n!!\n");
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
    acc_t * Tile_C = CUTE_result[CUTE_result_index];
    acc_t * Tile_D = D;


    //后操作的函数指针，返回值是void
    
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */
    MATMUL_MARCO_ISSUE();

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
        MATMUL_MARCO_SEARCH();

        printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*Tile_J+j,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A;
        Tile_B = B + j * 64 * stride_B;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = D;
        /*
        cute 配置
        cute 指令发射
        */
       MATMUL_MARCO_ISSUE();
        
        //执行当前任务的CPU的向量后操作任务
        printf("pre_i:%d,pre_j:%d\n",pre_i,pre_j);
        afater_operation(CUTE_result[CUTE_result_index],64,64,(C+pre_i*64*stride_C+pre_j*64),scale,stride_C);
        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*DIM_J+j,DIM_K);
        printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
        pre_j = j;
    }
    printf("[Final]pre_i:%d,pre_j:%d\n",pre_i,pre_j);
    afater_operation(CUTE_result[CUTE_result_index],64,64,(C+pre_i*64*stride_C+pre_j*64),scale,stride_C);
    printf("[Final][Vec]Vector Operation %s Finish\n",activation_name(act));
    

  }else
  {
    int Tile_I = DIM_I / 64;
    // int Tile_J = DIM_J / 64;

    int Application_M = 64;
    int Application_N = DIM_J;
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
    
    // afater_operation = act == SOFTMAX ? softmax_after_operation : NULL;
    

    //发射第一个CUTE的矩阵乘任务
    /*
    cute 配置
    cute 指令发射
    */
   MATMUL_MARCO_ISSUE();

    int i = 1;
    int pre_i = 0;

    int acc_not_finish = 1;
    volatile int acc_finish = 0;
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
        MATMUL_MARCO_SEARCH();

        printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*%d*%d\n",i,DIM_J,DIM_K);
        //发射下一个CUTE的矩阵乘任务
        Tile_A = A + i * 64 * stride_A;
        Tile_B = B;
        Tile_C = CUTE_result[CUTE_result_index==0?1:0];
        Tile_D = D;
        /*
        cute 配置
        cute 指令发射
        */
       MATMUL_MARCO_ISSUE();

        
        //执行当前任务的CPU的向量后操作任务
        printf("pre_i:%d:%d\n",pre_i);
        afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
        // printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*DIM_J+j,DIM_K);
        printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }
    printf("[Final]pre_i:%d\n",pre_i);
    afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
    printf("[Final][Vec]Vector Operation %s Finish\n",activation_name(act));
    
  }
}



static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type,int cute_transpose_B) {

    matmul_cute(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias,cute_transpose_B);
}

// Note: For self-attention, "enc_out" should be the same as "input".
// Note: "compression_factor" should be 1 for most use cases.
void attention(int hidden_dim, int expansion_dim, int num_heads, int seq_len,
        int compression_factor,

        const elem_t * input, const elem_t * enc_out,
        elem_t * out, elem_t * resadd_out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf)
{
    const int hidden_dim_compressed = hidden_dim / compression_factor;
    const int hidden_dim_per_head = hidden_dim_compressed / num_heads;

    // Q = Wq * input
    // K = Wk * enc_out
    // V = Wv * enc_out
    printf("[Hello YJP!]\n");
    const int qkv_matmuls_n = 3;
    for (int i = 0; i < qkv_matmuls_n; i++) {
        printf("[Embending!]\n");
        const elem_t * qkv_weights[] = {Wq, Wk, Wv};
        const elem_t * qkv_ins[] = {input, enc_out, enc_out};
        elem_t * qkv_outs[] = {Q_buf, K_buf, V_buf};

        const elem_t * qkv_w = qkv_weights[i];
        const elem_t * qkv_in = qkv_ins[i];
        elem_t * qkv_out = qkv_outs[i];

        tiled_matmul_auto(seq_len, hidden_dim_compressed, hidden_dim,
            /*A=*/ qkv_in, /*B=*/ qkv_w,
            /*D=*/ NULL, /*C=*/ qkv_out,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            CPU,i==1);
    }

    gemmini_fence();

    // attn = Q * K
    // attn = softmax(attn)
    for (int head = 0; head < num_heads; head++) {
        printf("[Get Attention! With SoftMax]\n");
        const elem_t * A = Q_buf + head * hidden_dim_per_head;
        const elem_t * B = K_buf + head * hidden_dim_per_head;
        elem_t * C = attn_buf + head * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, hidden_dim_per_head,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            /*SOFTMAX*/ LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ true,
            false, false,
            0,
            CPU,0);
    }

    gemmini_fence();

    // out_buf = attn * V
    for (int head = 0; head < num_heads; head++) {
        printf("[Get Attention's Out!]\n");
        const elem_t * A = attn_buf + head * seq_len * seq_len;
        const elem_t * B = V_buf + head * hidden_dim_per_head;
        elem_t * C = out_buf + head * hidden_dim_per_head;

        tiled_matmul_auto(seq_len, hidden_dim_per_head, seq_len,
            /*A=*/ A, /*B=*/ B,
            /*D=*/ NULL, /*C=*/ C,
            /*stride_A=*/seq_len, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
            /*repeating_bias=*/ false,
            false, /*transpose_B=*/ false,
            false, false,
            0,
            CPU,0);
    }

    gemmini_fence();

    // out = out_buf * Wo
    // out = LN(out)
    printf("[Get Out With LayerNrom!]\n");
    tiled_matmul_auto(seq_len, hidden_dim, hidden_dim_compressed,
        /*A=*/ out_buf, /*B=*/ Wo,
        /*D=*/ NULL, /*C=*/ out,
        /*stride_A=*/hidden_dim, /*stride_B=*/hidden_dim, /*stride_D=*/0, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ false,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU,0);

    gemmini_fence();

    // input = out + input
    printf("[Residual Add!]\n");
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        input,
        out,
        resadd_out,
        /*relu=*/ false,
        CPU);

    gemmini_fence();
}

void ffn(int hidden_dim, int expansion_dim, int seq_len,
        const elem_t * input, elem_t * out,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * out_buf)
{
    // out = FF1(input)
    // out = GELU(out)
    printf("[FF1!With Igelu]\n");
    tiled_matmul_auto(seq_len, expansion_dim, hidden_dim,
        /*A=*/ input, /*B=*/ ff1_w,
        /*D=*/ ff1_b, /*C=*/ out_buf,
        /*stride_A=*/hidden_dim, /*stride_B=*/expansion_dim, /*stride_D=*/expansion_dim, /*stride_C=*/expansion_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        IGELU, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ ACC_SCALE_IDENTITY,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU,0);

    gemmini_fence();

    // out = FF2(out)
    // out = LN(out)

    printf("[FF2!With LayerNorm]\n");
    tiled_matmul_auto(seq_len, hidden_dim, expansion_dim, 
        /*A=*/ out_buf, /*B=*/ ff2_w,
        /*D=*/ ff2_b, /*C=*/ out,
        /*stride_A=*/expansion_dim, /*stride_B=*/hidden_dim, /*stride_D=*/expansion_dim, /*stride_C=*/hidden_dim,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        LAYERNORM, /*scale=*/ ACC_SCALE_IDENTITY, /*bert_scale=*/ 0,
        /*repeating_bias=*/ true,
        false, /*transpose_B=*/ false,
        false, false,
        0,
        CPU,0);

    gemmini_fence();

    // out = out + input
    printf("[Residual Add!]\n");
    tiled_resadd_auto(seq_len, hidden_dim,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        out,
        input,
        out,
        /*relu=*/ false,
        CPU);

    gemmini_fence();
}

// Note: If "enc_out == NULL", then this will act as an encoder layer.
//   Otherwise, it will act as a decoder layer. If this is an encoder layer,
//   then "cross_num_heads" and all the "W*_cross" args are ignored.
uint64_t encoder_decoder(
        int hidden_dim, int expansion_dim, int num_heads, int cross_num_heads,
        int seq_len, int compression_factor,

        const elem_t * input, const elem_t * enc_out, elem_t * out,
        const elem_t * Wq, const elem_t * Wk, const elem_t * Wv, const elem_t * Wo,
        const elem_t * Wq_cross, const elem_t * Wk_cross, const elem_t * Wv_cross, const elem_t * Wo_cross,
        const elem_t * ff1_w, const elem_t * ff2_w,
        const acc_t * ff1_b, const acc_t * ff2_b,

        elem_t * Q_buf, elem_t * K_buf, elem_t * V_buf,
        elem_t * attn_buf, elem_t * out_buf,
        elem_t * resadd1_buf, elem_t * resadd2_buf)
{
    const bool is_encoder = enc_out == NULL;

    // uint64_t start = read_cycles();

    attention(hidden_dim, expansion_dim, num_heads, seq_len, compression_factor,
        input, input,
        out, resadd1_buf,
        Wq, Wk, Wv, Wo,
        Q_buf, K_buf, V_buf,
        attn_buf, out_buf);

    if (!is_encoder) {
        attention(hidden_dim, expansion_dim, cross_num_heads, seq_len, compression_factor,
            resadd1_buf, enc_out,
            out, resadd2_buf,
            Wq_cross, Wk_cross, Wv_cross, Wo_cross,
            Q_buf, K_buf, V_buf,
            attn_buf, out_buf);
    }

    ffn(hidden_dim, expansion_dim, seq_len,
        is_encoder ? resadd1_buf : resadd2_buf,
        out,
        ff1_w, ff2_w,
        ff1_b, ff2_b,
        out_buf);

    // uint64_t end = read_cycles();

    return 0;
}

#define ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, enc_out, output) ({ \
    static const elem_t Wqkvo[4][hidden_dim][hidden_dim]; \
    static const elem_t Wqkvo_cross[4][hidden_dim][hidden_dim]; \
    static const elem_t ff_w[2][hidden_dim*expansion_dim]; \
    static const acc_t ff1_b[expansion_dim]; \
    static const acc_t ff2_b[hidden_dim]; \
    \
    static elem_t QKV_buf[3][seq_len][hidden_dim];\
    static elem_t attn_buf[num_heads][seq_len][seq_len];\
    static elem_t out_buf[seq_len][expansion_dim];\
    static elem_t resadd1_buf[seq_len][hidden_dim];\
    static elem_t resadd2_buf[seq_len][hidden_dim];\
    \
    uint64_t cycles = encoder_decoder( \
            hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, \
            compression_factor, \
            \
            input, enc_out, output, \
            Wqkvo[0], Wqkvo[1], Wqkvo[2], Wqkvo[3],\
            Wqkvo_cross[0], Wqkvo_cross[1], Wqkvo_cross[2], Wqkvo_cross[3],\
            ff_w[0], ff_w[1], \
            ff1_b, ff2_b, \
            \
            QKV_buf[0], QKV_buf[1], QKV_buf[2], \
            attn_buf, out_buf, \
            resadd1_buf, resadd2_buf \
    ); \
    \
    cycles; \
})

#define PRINT_ENCODER_DECODER(name, is_encoder, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor) { \
    static const elem_t input[seq_len][hidden_dim]; \
    static const elem_t enc_out[seq_len][hidden_dim]; \
    static elem_t output[seq_len][hidden_dim]; \
    \
    char * type_str = is_encoder ? "encoder" : "decoder"; \
    \
    uint64_t cycles = ENCODER_DECODER(hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor, input, is_encoder ? NULL : enc_out, output); \
    \
    printf("%s stats: %s, hidden_dim=%d, expansion_dim=%d, num_heads=%d, cross_num_heads=%d, seq_len=%d, compression_factor=%d\n", \
            name, type_str, hidden_dim, expansion_dim, num_heads, cross_num_heads, seq_len, compression_factor); \
    printf("%s cycles: %llu\n\n", name, cycles); \
}

int main (int argc, char * argv[]) {

    // gemmini_flush(0);

    PRINT_ENCODER_DECODER("transformer-small", /*is_encoder=*/true,
            /*hidden_dim=*/512, /*expansion_dim=*/1024, /*num_heads=*/4, /*cross_num_heads=*/4, /*seq_len=*/128, /*compression_factor=*/1);

    PRINT_ENCODER_DECODER("bert-base", /*is_encoder=*/true,
            /*hidden_dim=*/768, /*expansion_dim=*/3072, /*num_heads=*/12, /*cross_num_heads=*/12, /*seq_len=*/128, /*compression_factor=*/1);

    // exit(0);
}

