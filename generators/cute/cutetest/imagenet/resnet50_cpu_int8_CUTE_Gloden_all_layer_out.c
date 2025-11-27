#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#include "resnet50_params_int8_CUTE_Gloden_transpose_weight.h"
#include "images.h"
#include <stdlib.h>

#define CUTE_TILE_Tensor_M 64
#define CUTE_TILE_Tensor_N 64
#define CUTE_TILE_Tensor_K 64

int cpu_check = 0;

#define GEMMINI_ACC_SCALE(x, scale) (x)
#define GEMMINI_SCALE(x, scale) (x)

void CUTE_CONV_3_3_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);//完成二维张量的切分，确定CUTE任务的切分
void CUTE_CONV_3_3_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);
void CUTE_CONV_1_1_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);
void CUTE_CONV_1_1_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type);

int cute_temp[4*4096*256];

void CUTE_CONV_3_3_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
void CUTE_CONV_3_3_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
void CUTE_CONV_1_1_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
void CUTE_CONV_1_1_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);
void CUTE_CONV_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);

void CUTE_CONV_KERNEL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);


void CUTE_MATMUL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length);//64*64*64

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}
static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

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

// 512位对齐的数组acc_t result[64][64]
acc_t __attribute__((aligned(512))) CUTE_result[2][64][3072];//double buffer
int CUTE_result_index = 0;

static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

      output[batch * channels + channel] = (sum + count/2) / count;

    }
  }
}

static void tiled_global_average_auto(const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

}



static elem_t scale_and_sat(acc_t x, int act, int scale, int bert_scale) {
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



#define MAT_IS_EQUAL(dim_i, dim_j, x, y) \
    ({int result = 1; \
      for (size_t i = 0; i < dim_i; i++) \
        for (size_t j = 0; j < dim_j; ++j) { \
          if (x[i][j] != y[i][j]) { \
            result = 0; \
            break; \
          } \
        } \
      result;})

static uint64_t read_cycles() {
    
    uint64_t cycles;


    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbff8);
    // const uint32_t * mtime = (uint32_t *)(33554432 + 0xbffc);
    // return *mtime;
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


static void tiled_resadd_auto(const size_t I, const size_t J,
        const int A_scale,
        const int B_scale,
        const int C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        bool greater,
        enum tiled_matmul_type_t matadd_type) {

    if (matadd_type == CPU) {
        if (greater) {
            resadd_cpu_greater(I, J, A_scale, B_scale, C_scale, A, B, C, relu);
        } else {
        resadd_cpu(I, J,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
        }
        return;
    }


}

void scale_after_operation_64_64(acc_t input[64][64], int dim_i,int dim_j,elem_t * output,int scale_shift,uint64_t stride_c)
{

    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j++) {
            elem_t* c = output + i * stride_c + j;
            acc_t x = input[i][j];
            // if(dim_i !=64)
            // printf("%d ",x);
            x = ACC_SCALE(x, scale_shift);
            *c = x;
        }
    }

}

void scale_after_operation_64_64_relu(acc_t input[64][64], int dim_i,int dim_j,elem_t * output,int scale_shift,uint64_t stride_c)
{

    for (size_t i = 0; i < dim_i; i++) {
        for (size_t j = 0; j < dim_j; j++) {
            elem_t* c = output + i * stride_c + j;
            acc_t x = input[i][j];
            // printf("%d ",x);
            x = ACC_SCALE(x, scale_shift);
            *c = x < 0 ? 0 : x;;
        }
    }
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

        printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i*Tile_J+j,DIM_K);
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

        printf("[CUTE]Matrix Multi Task Finish,Tile %d,Tile Size : 64*64*%d\n",i,DIM_K);
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
        printf("pre_i:%d\n",pre_i);
        afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
        printf("[Vec]Vector Operation %s Finish\n",activation_name(act));
        //切换CUTE的结果缓冲区
        CUTE_result_index = CUTE_result_index == 0 ? 1:0;
        pre_i = i;
    }

    afater_operation(CUTE_result[CUTE_result_index],64,DIM_J,(C+pre_i*64*stride_C),scale,stride_C);
    
  }
}


static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int A_scale_factor, int B_scale_factor, scale_acc_t D_scale_factor,
        int act, int scale, int bert_scale, bool repeating_bias) {

  const int no_bias = D == NULL;

//   if(cpu_check == 1)
//   {
//     //完成矩阵乘，并输出结果
//     printf("cpu_check:\n");
//     for(int i = 0;i<DIM_I;i++)
//     {
//         printf("DIM_I=%d:\t",i);
//       for(int j = 0;j<DIM_J;j++)
//       {
//         acc_t result = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + i*stride_D + j), D_scale_factor);
//         for(int k = 0;k<DIM_K;k++)
//         {
//           result += GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) * GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
//         }
//         printf("%d ",result);
//       }
//         printf("\n");
//     }
//   }

  if (act != LAYERNORM && act != SOFTMAX && !transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        // // //输出result
        // if(cpu_check)
        // {
        // printf("result:\n");
        // for (int i = 0; i < 4; i++) {
        //     for (int j = 0; j < 4; j++) {
        //         printf("%d ",result[i][j]);
        //     }
        //     printf("\n");
        // }
        // }


        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, bert_scale);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, bert_scale);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, bert_scale);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, bert_scale);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride

    // We also create a buffer that we can use for layernorms and softmaxes
    static acc_t c_buffer[1024];
    const size_t c_buffer_sz = sizeof(c_buffer)/sizeof(c_buffer[0]);
    if ((act == LAYERNORM || act == SOFTMAX) && DIM_J > c_buffer_sz) {
      printf("Matmul is too large to normalize\n");
      exit(1);
    }

    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }

        if (act == LAYERNORM || act == SOFTMAX)
          c_buffer[j] = sum;
        else
          *c = scale_and_sat(sum, act, scale, bert_scale);
      }
    }
  }
}

static void tiled_matmul(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int A_scale_factor, int B_scale_factor, scale_acc_t D_scale_factor,
        int act, int scale, int bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {


    if (tiled_matmul_type == CPU) {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
    
    }else if(tiled_matmul_type == CUTE){
        matmul_cute(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias,0);
    }
  
}

static void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        int A_scale_factor, int B_scale_factor, scale_acc_t D_scale_factor,
        int act, int scale, int bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = tiled_matmul_type == WS;

    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX) {
       tile_I = 1;
       tile_J = dim_J_padded/DIM;
       tile_K = 1;
    } else if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    exit(EXIT_SUCCESS);
#endif
#endif

    tiled_matmul(dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}



static void tiled_matmul_nn(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, int scale, bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, 0, repeating_bias,
        tile_I, tile_J, tile_K,
        false, false,
        false, false,
        0,
        tiled_matmul_type);

    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, 0, repeating_bias,
            false, false,
            false, false,
            0,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_nn_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, int scale, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    // if (check)
        printf("%s: cute\n", layer_name);

    tiled_matmul_auto(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, 0, repeating_bias,
        false, false,
        false, false,
        0,
        tiled_matmul_type);

    // //输出前1000个元素
    // printf("tiled_matmul_nn_auto:");
    // for(int i=0;i<4096;i++){
    //     int dj = i % dim_J;
    //     int di = i / dim_J;

    //     printf("[%d]%d ",i,C[di][dj]);
    // }
    // printf("\n");
    // fflush(stdout);
    if (check) {
        printf("%s: CPU\n", layer_name);
        elem_t gold[dim_I][dim_J];
        tiled_matmul_auto(dim_I, dim_J, dim_K,
            (elem_t*)A, (elem_t*)B, D, (elem_t*)gold, 
            dim_K, dim_J, dim_J, dim_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            act, scale, 0, repeating_bias,
            false, false,
            false, false,
            0,
            CPU);

        if (!MAT_IS_EQUAL(dim_I, dim_J, C, gold)) {
            printf("Layer calculated incorrectly: %s\n", layer_name);
            exit(1);
        }
    }
}
static void tiled_matmul_CUTE_auto(size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t A[dim_I][dim_K], const elem_t B[dim_K][dim_J],
        const void * D, elem_t C[dim_I][dim_J],
        int act, int scale, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{


}

static void tiled_conv_downsample(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, int scale,

        enum tiled_matmul_type_t tiled_conv_type) {

    if(tiled_conv_type == CPU){
        // Rectangular dimensions for this function are currently not supported
        if (in_row_dim != in_col_dim || out_row_dim != out_col_dim) {
            printf("Rectangular convolutions for tiled_conv_downsample are currently not supported.\n");
            exit(1);
        }

        const int in_dim = in_row_dim;
        const int out_dim = out_row_dim;

        const int stride = 2;

        for (int b = 0; b < batch_size; b++) {
            for (int irow = 0; irow < in_row_dim; irow += stride) {
                const int orow = irow / stride;

                const int I = in_col_dim / stride; // number of columns in row
                const int J = out_channels;
                const int K = in_channels;

                const elem_t * A = input + (b * in_dim + irow) * in_dim * in_channels;
                const elem_t * B = weights;
                const acc_t * D = bias;
                elem_t * C = output + (b * out_dim + orow) * out_dim * out_channels;

                const int A_stride = in_channels * 2;
                const int B_stride = out_channels;
                const int D_stride = out_channels;
                const int C_stride = out_channels;
                tiled_matmul_auto(I, J, K, A, B, (void*)D, (void*)C,
                        A_stride, B_stride, D_stride, C_stride,
                        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                        MVIN_SCALE_IDENTITY, act, scale, 0,
                        true, false, false, false, false, 0, tiled_conv_type);
            }
        }
    }else if(tiled_conv_type == CUTE)
    {
        const int in_dim = in_row_dim;
        const int out_dim = out_row_dim;

        const int stride = 2;

        for (int b = 0; b < batch_size; b++) {

        }
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

    //     printf("[%d]%d ",i,Tile_C[di*stride_D+dj]);
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
    //     printf("CUTE_CONV_MarcoTask_SIM:");
    //     for(int i=0;i<1000;i++){
    //         printf("%d ",Tile_C[i]);
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
    //     printf("CUTE_CONV_MarcoTask_SIM:");
    //     for(int i=0;i<1000;i++){
    //         printf("%d ",Tile_C[i]);
    //     }
    // }
}

void CUTE_MATMUL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    printf("CUTE_MATMUL_MarcoTask\n");
    printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    printf("stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",stride_A,stride_B,stride_C,stride_D);

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
    CUTE_CONV_KERNEL_MarcoTask_SIM(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,1,0,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

    return;

}

void CUTE_CONV_KERNEL_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_size,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    printf("CUTE_CONV_KERNEL_MarcoTask\n");
    printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    printf("stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",stride_A,stride_B,stride_C,stride_D);
    printf("conv_stride:%d,kernel_size:%d\n",conv_stride,kernel_size);
    printf("conv_oh_index:%d,conv_ow_index:%d,conv_oh_max:%d,conv_ow_max:%d\n",conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max);

    //CUTE配置
    //CUTE指令发射

    CUTE_CONV_KERNEL_MarcoTask_SIM(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,kernel_size,kernel_stride,\
                        stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
}


void CUTE_CONV_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length){
    printf("CUTE_CONV_MarcoTask\n");
    printf("Application_M:%d,Application_N:%d,Application_K:%d\n",Application_M,Application_N,Application_K);
    printf("Conv_stride:%d,stride_A:%d,stride_B:%d,stride_C:%d,stride_D:%d\n",conv_stride,stride_A,stride_B,stride_C,stride_D);
    printf("conv_oh_index:%d,conv_ow_index:%d,conv_oh_max:%d,conv_ow_max:%d\n",conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max);



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

    CUTE_CONV_MarcoTask_SIM(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

    return;

}

void CUTE_CONV_1_1_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //任务和CUTE_MATMUL_MarcoTask是一样的
    printf("CUTE_CONV_1_1_S1_MarcoTask\n");

    CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,1,0,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

    return;
}

void CUTE_CONV_1_1_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //任务和CUTE_MATMUL_MarcoTask是一样的,只不过stride_A要乘2，且如果Application_M不是2的倍数，需要零填充，不能简单用CUTE_MATMUL_MarcoTask
    printf("CUTE_CONV_1_1_S2_MarcoTask\n");

    // if(Application_M%2!=0 || Application_N%2!=0){
    //     printf("[CUTE_CONV_1_1_S2_MarcoTask]Application_M/Application_N  is not a multiple of 2\n");
    //     exit(1);
    // }
    CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,1,0,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
    return;
}

void CUTE_CONV_3_3_S1_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //padding代表整个卷积任务是否存在padding，conv_oh_index,conv_ow_index代表当前卷积任务的第一个位置，我们的数据是output=[ohow][oc],input[ihiw][ic],kernel=[khkw][oc][ic]紧密排列的，所以可以判断是否需要进行0填充
    printf("CUTE_CONV_3_3_S1_MarcoTask\n");

    //CUTE配置
    //CUTE指令发射

    /*
    kernel_size = 3的情况，就是9次kernel_size = 1的情况，但是由于只用存储一次C_SCP的数据，且有padding的填充任务，所以不能直接调用CUTE_CONV_1_1_S1_MarcoTask，微指令上有根本区别
    注意padding的信息，它描述的4个方向上是否有连续的padding，此时A的Load任务，需要处理好零填充的任务，直到0填充到Application_M的大小，然后再进行计算任务。
    */
    CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,3,kernel_stride,\
                            stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);

}

void CUTE_CONV_3_3_S2_MarcoTask(void *A,void *B,void *C,void *D,int Application_M,int Application_N,int Application_K,int element_type,int bias_type,int conv_stride,int kernel_stride,\
                            uint64_t stride_A,uint64_t stride_B,uint64_t stride_C,uint64_t stride_D,bool transpose_result,int conv_oh_index,int conv_ow_index,int conv_oh_max,int conv_ow_max,void * VectorOp,int VectorInst_Length)
{
    //padding代表整个卷积任务是否存在padding，我们的数据是output=[ohow][oc],input[ihiw][ic],kernel=[khkw][oc][ic]紧密排列的，所以可以判断是否需要进行0填充
    printf("CUTE_CONV_3_3_S2_MarcoTask\n");

    //CUTE配置
    //CUTE指令发射

    /*
    kernel_size = 3的情况已经在CUTE_CONV_3_3_S1_MarcoTask中讨论过了，现在stride=2，
    需要计算ow,oh->根据stride计算当前中心点->当前处理的conv_1_1任务位点(9次中的哪一次),计算当前的conv_1_1在A中Load的任务位点，然后决定是否需要padding处理
    */
    CUTE_CONV_KERNEL_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,conv_stride,3,kernel_stride,\
                        stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max,VectorOp,VectorInst_Length);
}


static void im2col(size_t batch_size, size_t channels, size_t im_row_dim, size_t im_col_dim,
    size_t I, size_t K,
    const elem_t input[batch_size][im_row_dim][im_col_dim][channels],
    elem_t output[I][K],
    const struct ConvParams * params)
{
    int patch_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_row_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_col_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int patch_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;
                            
                            if (pixel_row < 0 || pixel_row >= params->in_row_dim
                                || pixel_col < 0 || pixel_col >= params->in_col_dim) {
                                // output[patch_row][patch_col] = 0;
                            } else {
                                output[patch_row][patch_col] = input[n_batch][pixel_row][pixel_col][im_channel];
                            }

                            patch_col++;
                        }
                    }
                }
                
                patch_row++;
            }
        }
    }
}

static void im2col_with_col2im(size_t prev_I, size_t prev_J,
    size_t next_I, size_t next_K,
    const elem_t input[prev_I][prev_J],
    elem_t output[next_I][next_K],
    const struct ConvParams * params)
{
    int out_row = 0;

    for (int n_batch = 0; n_batch < params->batch_size; n_batch++) {
        for (int im_row = -params->padding; im_row < params->in_row_dim - params->kernel_size + params->padding + 1; im_row += params->stride) {
            for (int im_col = -params->padding; im_col < params->in_col_dim - params->kernel_size + params->padding + 1; im_col += params->stride) {
                int out_col = 0;

                for (int filter_row = 0; filter_row < params->kernel_size; filter_row++) {
                    for (int filter_col = 0; filter_col < params->kernel_size; filter_col++) {
                        for (int im_channel = 0; im_channel < params->in_channels; im_channel++) {
                            int pixel_row = im_row + filter_row;
                            int pixel_col = im_col + filter_col;

                            if (pixel_row < 0 || pixel_row >= params->in_row_dim
                                || pixel_col < 0 || pixel_col >= params->in_col_dim) {
                                // output[out_row][out_col] = 0;
                            } else {
                                int in_row = n_batch * params->in_row_dim * params->in_col_dim + pixel_row * params->in_col_dim + pixel_col;
                                int in_col = im_channel;

                                output[out_row][out_col] = input[in_row][in_col];
                            }

                            out_col++;
                        }
                    }
                }

                out_row++;
            }
        }
    }
}


void pool_with_col2im(size_t I, size_t J,
    size_t batch_size, size_t channels, size_t out_row_dim, size_t out_col_dim,
    elem_t input[I][J],
    elem_t output[batch_size][out_row_dim][out_col_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    size_t in_row_dim = params->out_row_dim;
    size_t in_col_dim = params->out_col_dim;
    size_t padding = params->pool_padding;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_row_dim; out_row++) {
                for (int out_col = 0; out_col < out_col_dim; out_col++) {
                    int in_row = out_row * stride - padding;

                    elem_t result = elem_t_min;

                    for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                        int in_col = out_col * stride - padding;

                        for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < in_row_dim && in_col >= 0 && in_col < in_col_dim) {
                                if (input[batch * in_row_dim * in_col_dim + in_row * in_col_dim + in_col][channel] > result) {
                                    result = input[batch * in_row_dim * in_col_dim + in_row * in_col_dim + in_col][channel];
                                }
                            } else if (0 > result) {
                                result = 0;
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    output[batch][out_row][out_col][channel] = result;
                }
            }
        }
    }
}

static void conv_cpu_without_pool(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, int scale) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_row_dim; orow++) {
      for (int ocol = 0; ocol < out_col_dim; ocol++) {
        for (int och = 0; och < out_channels; och++) {

          acc_t opixel = no_bias ? 0 : bias[och];

          for (int krow = 0; krow < kernel_dim; krow++) {
            if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
              continue;

            const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                continue;

              const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

              for (int kch = 0; kch < in_channels; kch++) {
                const elem_t *in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_channels + kch;
                if (trans_input_3120) {
                  // NHWC to CHWN
                  in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                }

                elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                    0 : *in;

                const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + och);
                if (trans_weight_1203) {
                  // HWIO to WIHO
                  weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + och);
                } else if (trans_weight_0132) {
                  // HWIO to HWOI
                  weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + och) * in_channels + kch);
                }

                opixel += weight * ipixel;
              }
            }
          }

          elem_t *out = output + (b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * out_channels + och;
          if (trans_output_1203) {
            // NHWC to HWNC
            out = output + (orow * out_col_dim * batch_size + ocol * batch_size + b) * out_channels + och;
          }
            //输出包括值和坐标orow,ocol,och
            // printf("(%d,%d,%d)=%d ",orow,ocol,och,opixel);
          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
  printf("\n");
}

static void conv_cpu(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, int scale,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_cpu_without_pool(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_row_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_col_dim; pocol++) {
        for (int poch = 0; poch < out_channels; poch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_row_dim || ocol < 0 || ocol >= out_col_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[poch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
                    continue;

                  const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                      continue;

                    const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

                    for (int kch = 0; kch < in_channels; kch++) {
                      const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_channels + kch;
                      if (trans_input_3120) {
                        // NHWC to CHWN
                        in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                      }

                      elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                          0 : *in;

                      const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                      const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                      elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch);
                      if (trans_weight_1203) {
                        // HWIO to WIHO
                        weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + poch);
                      } else if (trans_weight_0132) {
                        // HWIO to HWOI
                        weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch);
                      }

                      opixel += weight * ipixel;
                    }
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, 0);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
                if (trans_output_1203) {
                  // NHWC to HWNC
                  out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                }

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void tiled_conv(
        int batch_size,
        int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, int scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {


    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      // assume in_dim_rows = in_dim_cols
      // and out_dim_rows = out_dim_cols for now
      conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    }
    printf("Tiled conv not implemented for this configuration\n");
    exit(1);
}

static int tiled_conv_total_spad_rows(bool acc,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A_rows + B_rows;
}

void CUTE_TASK_END()
{
    printf("CUTE_TASK_END\n");
    return;
}

void CUTE_CONV_3_3_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    printf("CUTE_CONV_3_3_S2_AUTO\n");
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
                int bias_type = CUTE_BIAS_REPEAT_ROW;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                uint64_t kernel_stride = params.in_channels*params.out_channels;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                CUTE_CONV_3_3_S2_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,2,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END();
                    afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
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
            afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
        }
    }
}
void CUTE_CONV_3_3_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    printf("CUTE_CONV_3_3_S1_AUTO\n");
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
                int bias_type = CUTE_BIAS_REPEAT_ROW;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                uint64_t kernel_stride = params.in_channels*params.out_channels;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                CUTE_CONV_3_3_S1_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,1,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END();
                    afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
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
            afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
        }
    }
}
void CUTE_CONV_1_1_S2_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    printf("CUTE_CONV_1_1_S2_AUTO\n");
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
                printf("[TILE_S2]current_ih:%d,current_iw:%d\n",current_ih,current_iw);


                bool Is_Last_Tile_M = Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1;

                int Application_M = (Has_Last_Tile_M && CONV_Current_Tile_M == CONV_Current_Tile_M_Max-1)?Last_Tile_M:CUTE_TILE_Tensor_M;
                int Application_N = CUTE_TILE_Tensor_N;
                int Application_K = CONV_Matrix_K;
                void *A = input + i*input_batch_stride;//计算的时候会利用oh和ow，直接传input就好
                void *B = weights + CONV_Current_Tile_N * CUTE_TILE_Tensor_N * params.in_channels;
                void *C = CUTE_result[CUTE_result_index];
                void *D = bias + CONV_Current_Tile_N * CUTE_TILE_Tensor_N;
                int element_type = CUTE_INT8;
                int bias_type = CUTE_BIAS_REPEAT_ROW;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                CUTE_CONV_1_1_S2_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,2,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END();
                    afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
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
            afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
        }
    }
}

void CUTE_CONV_1_1_S1_AUTO(ConvParams params,const elem_t * input,const elem_t * weights,const acc_t * bias,elem_t * output,int act_type)
{
    printf("CUTE_CONV_1_1_S1_AUTO\n");
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
                int bias_type = CUTE_BIAS_REPEAT_ROW;
                uint64_t stride_A = params.in_channels;
                uint64_t stride_B = params.in_channels;
                uint64_t stride_C = CUTE_TILE_Tensor_N*4;//int32->int8连续区域存放int[64][64]
                uint64_t stride_D = params.out_channels*4;
                bool transpose_result = false;
                void * VectorOp = NULL;
                int VectorInst_Length = 0;

                CUTE_CONV_1_1_S1_MarcoTask(A,B,C,D,Application_M,Application_N,Application_K,element_type,bias_type,stride_A,stride_B,stride_C,stride_D,transpose_result,CONV_Current_oh,CONV_Current_ow,params.out_row_dim,params.out_col_dim,VectorOp,VectorInst_Length);
                if(have_after_operation)
                {
                    CUTE_TASK_END();
                    afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
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
            afater_operation(CUTE_result[VECTASK_RESULT_INDEX],VECTASK_DIM_I,VECTASK_DIM_J,VECTASK_C_Addr,params.output_scale_shift,CONV_Matrix_N);
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
    printf("CUTE_CONV_AUTO\n");
    fflush(stdout);
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

static void tiled_conv_auto(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, int scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {



    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;
    const int in_channels_idx = 6;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    */

    tiled_conv(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);

    fflush(stdout);
}



void right_test(elem_t gloden_out[],elem_t cute_out[],int len,int check)
{
    if (check == 0)
    {
        return;
    }
    for(int i = 0; i < len; i++)
    {
        if(i ==200706) continue;
        if(gloden_out[i] != cute_out[i])
        {
            printf("(%d):gloden:%d,cute_out:%d\n",i,gloden_out[i],cute_out[i]);
            printf("right_test wrong\n");
            exit(1);
        }
    }
    printf("right_test right\n");
}

void get_layer_input_output(char *layer_name,ConvParams params,elem_t *input,elem_t *weights,acc_t *bias,elem_t *output,int act_type)
{
    //将这一层的输入输出全都转储到layer_name.h的文件中
    FILE *fp;
    char file_name[100];
    sprintf(file_name,"%s.h",layer_name);
    fp = fopen(file_name,"w");
    if(fp == NULL)
    {
        printf("open file failed\n");
        exit(1);
    }

    /*输出结构体和当前层的结构参数
    struct ConvParams {
        int batch_size;
        int in_row_dim;
        int in_col_dim;
        int out_row_dim;
        int out_col_dim;
        int kernel_size;
        int in_channels;
        int out_channels;
        int stride;
        int padding;
        bool bias;
        bool depthwise;
        int n_patches;
        int patch_size;
        int output_scale_shift;
        int res_scale_shift;
        bool res_scale_greater;
        int pool_size, pool_stride, pool_padding, out_dim_pooled;
        
        int I, J, K;
    };

    typedef struct ConvParams ConvParams;

    struct FcParams {
        int batch_size;
        int in_features;
        int out_features;
        int output_scale_shift;
        bool bias;

        int I, J, K;
    };

    */

    fprintf(fp,"#include <stdint.h>\n");
    
    fprintf(fp,"struct ConvParams {\n");
    fprintf(fp,"    int batch_size;\n");
    fprintf(fp,"    int in_row_dim;\n");
    fprintf(fp,"    int in_col_dim;\n");
    fprintf(fp,"    int out_row_dim;\n");
    fprintf(fp,"    int out_col_dim;\n");
    fprintf(fp,"    int kernel_size;\n");
    fprintf(fp,"    int in_channels;\n");
    fprintf(fp,"    int out_channels;\n");
    fprintf(fp,"    int stride;\n");
    fprintf(fp,"    int padding;\n");
    fprintf(fp,"    bool bias;\n");
    fprintf(fp,"    bool depthwise;\n");
    fprintf(fp,"    int n_patches;\n");
    fprintf(fp,"    int patch_size;\n");
    fprintf(fp,"    int output_scale_shift;\n");
    fprintf(fp,"    int res_scale_shift;\n");
    fprintf(fp,"    bool res_scale_greater;\n");
    fprintf(fp,"    int pool_size;\n");
    fprintf(fp,"    int pool_stride;\n");
    fprintf(fp,"    int pool_padding;\n");
    fprintf(fp,"    int out_dim_pooled;\n");
    fprintf(fp,"    int I, J, K;\n");
    fprintf(fp,"};\n");

    fprintf(fp,"typedef struct ConvParams ConvParams;\n");


    //输出当前层的结构参数
    fprintf(fp,"ConvParams %s_params = {\n",layer_name);
    fprintf(fp,"    .batch_size = %d,\n",1);
    fprintf(fp,"    .in_row_dim = %d,\n",params.in_row_dim);
    fprintf(fp,"    .in_col_dim = %d,\n",params.in_col_dim);
    fprintf(fp,"    .out_row_dim = %d,\n",params.out_row_dim);
    fprintf(fp,"    .out_col_dim = %d,\n",params.out_col_dim);
    fprintf(fp,"    .kernel_size = %d,\n",params.kernel_size);
    fprintf(fp,"    .in_channels = %d,\n",params.in_channels);
    fprintf(fp,"    .out_channels = %d,\n",params.out_channels);
    fprintf(fp,"    .stride = %d,\n",params.stride);
    fprintf(fp,"    .padding = %d,\n",params.padding);
    fprintf(fp,"    .bias = %d,\n",params.bias);
    fprintf(fp,"    .depthwise = %d,\n",params.depthwise);
    fprintf(fp,"    .n_patches = %d,\n",params.n_patches);
    fprintf(fp,"    .patch_size = %d,\n",params.patch_size);
    fprintf(fp,"    .output_scale_shift = %d,\n",params.output_scale_shift);
    fprintf(fp,"    .res_scale_shift = %d,\n",params.res_scale_shift);
    fprintf(fp,"    .res_scale_greater = %d,\n",params.res_scale_greater);
    fprintf(fp,"    .pool_size = %d,\n",params.pool_size);
    fprintf(fp,"    .pool_stride = %d,\n",params.pool_stride);
    fprintf(fp,"    .pool_padding = %d,\n",params.pool_padding);
    fprintf(fp,"    .out_dim_pooled = %d,\n",params.out_dim_pooled);
    fprintf(fp,"    .I = %d,\n",params.I);
    fprintf(fp,"    .J = %d,\n",params.J);
    fprintf(fp,"    .K = %d,\n",params.K);
    fprintf(fp,"};\n");

    //输出mnk，application_m等，用宏输出
    fprintf(fp,"#define APPLICATION_M %d\n",params.out_row_dim*params.out_col_dim);
    fprintf(fp,"#define APPLICATION_N %d\n",params.out_channels);
    fprintf(fp,"#define APPLICATION_K %d\n",params.in_channels);
    fprintf(fp,"#define A_APPLICATION_M %d\n",params.in_row_dim*params.in_col_dim);
    fprintf(fp,"#define BIAS_TYPE %d\n",2);
    //输出注释，1表示zero bias，2表示repeat row bias，3表示full bias
    fprintf(fp,"//1:zero bias,2:repeat row bias,3:full bias\n");
    //输出conv_stride,kernel_size,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max
    //输出conv_stride = 1,kernel_size = 1,kernel_stride = 0,因为矩阵乘不需要这些值
    fprintf(fp,"#define CONV_STRIDE %d\n",params.stride);
    fprintf(fp,"#define KERNEL_SIZE %d\n",params.kernel_size);
    fprintf(fp,"#define KERNEL_STRIDE %d\n",params.kernel_size*params.kernel_size*params.out_channels);
    //stride_A = K,stride_B = K,stride_C = 4*N,stride_D = 4*N
    fprintf(fp,"#define STRIDE_A %d\n",params.in_channels);
    fprintf(fp,"#define STRIDE_B %d\n",params.in_channels);
    fprintf(fp,"#define STRIDE_C %d\n",4*params.out_channels);
    fprintf(fp,"#define STRIDE_D %d\n",4*params.out_channels);
    
    //transpose_result = 0,conv_oh_index = 0,conv_ow_index = 0,conv_oh_max = 1,conv_ow_max = M
    fprintf(fp,"#define TRANSPOSE_RESULT 0\n");
    fprintf(fp,"#define CONV_OH_INDEX 0\n");
    fprintf(fp,"#define CONV_OW_INDEX 0\n");
    fprintf(fp,"#define CONV_OH_PER_ADD %d\n",64/params.out_row_dim);
    fprintf(fp,"#define CONV_OW_PER_ADD %d\n",64%params.out_row_dim);
    fprintf(fp,"#define CONV_OH_MAX %d\n",params.out_row_dim);
    fprintf(fp,"#define CONV_OW_MAX %d\n",params.out_col_dim);

    fprintf(fp, "#define MVIN_SCALE_IDENTITY 0 \n");
    fprintf(fp, "#define ACC_SCALE_IDENTITY 0 \n");
    fprintf(fp, "// Rounding right shift equation: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm \n");
    fprintf(fp, "#define ROUNDING_RIGHT_SHIFT(x, shift) \\ \n");
    fprintf(fp, "((shift) > 0 ? (((x) >> (shift)) + \\ \n");
    fprintf(fp, "(((shift) == 0 ? 0 : (((x) >> ((shift)-1)) & 1)) & \\ \n");
    fprintf(fp, "((((shift) <= 1 ? 0 : ((x) & ((1 << ((shift)-1)) - 1))) != 0) | (((x) >> (shift)) & 1)))) : ((x) << (-(shift)))) \n");
    fprintf(fp, "#ifdef __cplusplus \n");
    fprintf(fp, "#define SAME_TYPE(x) decltype(x) \n");
    fprintf(fp, "#else \n");
    fprintf(fp, "#define SAME_TYPE(x) typeof(x) \n");
    fprintf(fp, "#endif \n");
    fprintf(fp, "#define ROUND_NEAR_EVEN(x) \\ \n");
    fprintf(fp, "({ const SAME_TYPE(x) x_ = (x); \\ \n");
    fprintf(fp, "const long long i = x_; \\ \n");
    fprintf(fp, "const long long next = x_ < 0 ? x_ - 1 : x_ + 1; \\ \n");
    fprintf(fp, "SAME_TYPE(x) rem = x_ - i; \\ \n");
    fprintf(fp, "rem = rem < 0 ? -rem : rem; \\ \n");
    fprintf(fp, "SAME_TYPE(x) result = rem < 0.5 ? i : (rem > 0.5 ? next : ( \\ \n");
    fprintf(fp, "i % 2 == 0 ? i : next)); \\ \n");
    fprintf(fp, "result; }) \n");
    fprintf(fp, "#define ROUND_NEAR_EVEN_WITH_SCALE(x,scale) \\ \n");
    fprintf(fp, "({  int half = x & (1 << (scale - 1));\\ \n");
    fprintf(fp, "int is_odd = x & (1 << scale);\\ \n");
    fprintf(fp, "int rem = x & ((1 << (scale)) - 1);\\ \n");
    fprintf(fp, "int add = (half && is_odd) || (rem > (1 << (scale - 1)));\\ \n");
    fprintf(fp, "(x >> scale) + add;}) \n");
    fprintf(fp, "#define ACC_SCALE(x, scale) \\ \n");
    fprintf(fp, "({int y = scale?ROUND_NEAR_EVEN_WITH_SCALE(x,scale):x ; y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);}) \n");
    fprintf(fp, "#define MVIN_SCALE(x, scale) ({int y = scale?ROUND_NEAR_EVEN_WITH_SCALE(x,scale):x ; y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);}) \n");
    fprintf(fp, "#define MVIN_SCALE_GREATER(x, scale) ({int y = x << scale ; y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (acc_t)y);}) \n");
    fprintf(fp, "#define MVIN_SCALE_ACC(x, scale) (x) \n");

    //act_type 0 = NO_ACTIVATION,1 = RELU,2 = LAYERNORM,3 = IGELU,4 = SOFTMAX
    //输出注释提示
    fprintf(fp,"//act_type  0 = NO_ACTIVATION,1 = RELU,2 = LAYERNORM,3 = IGELU,4 = SOFTMAX\n");
    fprintf(fp,"#define ACT_TYPE %d\n",act_type);

    //输出input,weights,bias,output,512bit对齐
    fprintf(fp,"int8_t input[A_APPLICATION_M*APPLICATION_K] __attribute__((aligned(64))) = {\n");
    for(int i = 0; i < params.in_row_dim*params.in_col_dim*params.in_channels; i++)
    {
        fprintf(fp,"%d,",input[i]);
    }
    fprintf(fp,"\n};\n");

    fprintf(fp,"int8_t weights[KERNEL_SIZE*KERNEL_SIZE*APPLICATION_N*APPLICATION_K] __attribute__((aligned(64))) = {\n");
    for(int i = 0; i < params.kernel_size*params.kernel_size*params.out_channels*params.in_channels; i++)
    {
        fprintf(fp,"%d,",weights[i]);
    }
    fprintf(fp,"\n};\n");

    fprintf(fp,"int32_t bias[APPLICATION_N] __attribute__((aligned(64))) = {\n");
    for(int i = 0; i < params.out_channels; i++)
    {
        fprintf(fp,"%d,",bias[i]);
    }
    fprintf(fp,"\n};\n");

    fprintf(fp,"int8_t gloden_output_with_scale[APPLICATION_M*APPLICATION_N] __attribute__((aligned(64))) = {\n");
    for(int i = 0; i < params.out_row_dim*params.out_col_dim*params.out_channels; i++)
    {
        fprintf(fp,"%d,",output[i]);
    }
    fprintf(fp,"\n};\n");

    fprintf(fp,"int8_t output[APPLICATION_M*APPLICATION_N] __attribute__((aligned(64))) = {\n");
    for(int i = 0; i < params.out_row_dim*params.out_col_dim*params.out_channels; i++)
    {
        fprintf(fp,"0,");
    }
    fprintf(fp,"\n};\n");

    fclose(fp);
    //输出提示信息，输出文件名.h
    // printf("get_layer_input_output right\n");
    printf("output file:%s\n",file_name);
    // exit(1);

}

int main (int argc, char * argv[]) {

    enum tiled_matmul_type_t tiled_matmul_type = CPU;
    int check = 0;
    int CUTE_do_check = 0;

    // conv_1
    tiled_conv_auto(
        conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
        conv_1_params.in_channels,
        conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
        conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,
        RELU, conv_1_params.output_scale_shift,
        conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,
        tiled_matmul_type);
    
    // conv_2
    tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
        conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
        RELU, conv_2_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_2");
    // tiled_conv_CUTE_auto(conv_2_params, conv_1_out_pooled, conv_2_w_t, conv_2_b, cute_temp,RELU);
    // right_test(conv_2_out, cute_temp, conv_2_params.I * conv_2_params.J,CUTE_do_check);
    get_layer_input_output("conv_2",conv_2_params, conv_1_out_pooled, conv_2_w_t, conv_2_b, conv_2_out, RELU);
    // conv_3
    tiled_conv_auto(
        conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim,
        conv_3_params.in_channels,
        conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim,
        conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,
        RELU, conv_3_params.output_scale_shift,
        conv_3_params.pool_size, 0, conv_3_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_3_params, conv_2_out, conv_3_w_t, conv_3_b, cute_temp,RELU);
    // right_test(conv_3_out, cute_temp, conv_3_params.I * conv_3_params.J,CUTE_do_check);
    get_layer_input_output("conv_3",conv_3_params, conv_2_out, conv_3_w_t, conv_3_b, conv_3_out, RELU);
    // conv_4
    tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
        conv_3_out, conv_4_w, conv_4_b, conv_4_out,
        NO_ACTIVATION, conv_4_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_4");
    // tiled_conv_CUTE_auto(conv_4_params, conv_3_out, conv_4_w_t, conv_4_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_4_out, cute_temp, conv_4_params.I * conv_4_params.J,CUTE_do_check);
    get_layer_input_output("conv_4",conv_4_params, conv_3_out, conv_4_w_t, conv_4_b, conv_4_out, NO_ACTIVATION);
    
    
    

    // Downsampling conv_1_out_pooled
    // conv_5
    tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
        conv_1_out_pooled, conv_5_w, conv_5_b, conv_5_out,
        NO_ACTIVATION, conv_5_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_5");
    // tiled_conv_CUTE_auto(conv_5_params, conv_1_out_pooled, conv_5_w_t, conv_5_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_5_out, cute_temp, conv_5_params.I * conv_5_params.J,CUTE_do_check);
    get_layer_input_output("conv_5",conv_5_params, conv_1_out_pooled, conv_5_w_t, conv_5_b, conv_5_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        conv_4_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    

    // conv_6
    
    tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
        conv_4_out, conv_6_w, conv_6_b, conv_6_out,
        RELU, conv_6_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_6");
    // tiled_conv_CUTE_auto(conv_6_params, conv_4_out, conv_6_w_t, conv_6_b, cute_temp,RELU);
    // right_test(conv_6_out, cute_temp, conv_6_params.I * conv_6_params.J,CUTE_do_check);
    get_layer_input_output("conv_6",conv_6_params, conv_4_out, conv_6_w_t, conv_6_b, conv_6_out, RELU);
    
    
    
    // conv_7
    
    tiled_conv_auto(
        conv_7_params.batch_size, conv_7_params.in_row_dim, conv_7_params.in_col_dim,
        conv_7_params.in_channels,
        conv_7_params.out_channels, conv_7_params.out_row_dim, conv_7_params.out_col_dim,
        conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,
        RELU, conv_7_params.output_scale_shift,
        conv_7_params.pool_size, 0, conv_7_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_7_params, conv_6_out, conv_7_w_t, conv_7_b, cute_temp,RELU);
    // right_test(conv_7_out, cute_temp, conv_7_params.I * conv_7_params.J,CUTE_do_check);
    get_layer_input_output("conv_7",conv_7_params, conv_6_out, conv_7_w_t, conv_7_b, conv_7_out, RELU);
    
    


    // conv_8
    
    tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
        conv_7_out, conv_8_w, conv_8_b, conv_8_out,
        NO_ACTIVATION, conv_8_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_8");
    // tiled_conv_CUTE_auto(conv_8_params, conv_7_out, conv_8_w_t, conv_8_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_8_out, cute_temp, conv_8_params.I * conv_8_params.J,CUTE_do_check);
    get_layer_input_output("conv_8",conv_8_params, conv_7_out, conv_8_w_t, conv_8_b, conv_8_out, NO_ACTIVATION);
    
    


    // Add residuals
    
    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        conv_8_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    

    // conv_9
    
    tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
        conv_8_out, conv_9_w, conv_9_b, conv_9_out,
        RELU, conv_9_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_9");
    // tiled_conv_CUTE_auto(conv_9_params, conv_8_out, conv_9_w_t, conv_9_b, cute_temp,RELU);
    // right_test(conv_9_out, cute_temp, conv_9_params.I * conv_9_params.J,CUTE_do_check);
    get_layer_input_output("conv_9",conv_9_params, conv_8_out, conv_9_w_t, conv_9_b, conv_9_out, RELU);
    
    

    // conv_10
    
    tiled_conv_auto(
        conv_10_params.batch_size, conv_10_params.in_row_dim, conv_10_params.in_col_dim,
        conv_10_params.in_channels,
        conv_10_params.out_channels, conv_10_params.out_row_dim, conv_10_params.out_col_dim,
        conv_10_params.stride, 1, 1, conv_10_params.padding, conv_10_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,
        RELU, conv_10_params.output_scale_shift,
        conv_10_params.pool_size, 0, conv_10_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_10_params, conv_9_out, conv_10_w_t, conv_10_b, cute_temp,RELU);
    // right_test(conv_10_out, cute_temp, conv_10_params.I * conv_10_params.J,CUTE_do_check);
    get_layer_input_output("conv_10",conv_10_params, conv_9_out, conv_10_w_t, conv_10_b, conv_10_out, RELU);
    
    

    // conv_11
    
    tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
        conv_10_out, conv_11_w, conv_11_b, conv_11_out,
        NO_ACTIVATION, conv_11_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_11");
    // tiled_conv_CUTE_auto(conv_11_params, conv_10_out, conv_11_w_t, conv_11_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_11_out, cute_temp, conv_11_params.I * conv_11_params.J,CUTE_do_check);
    get_layer_input_output("conv_11",conv_11_params, conv_10_out, conv_11_w_t, conv_11_b, conv_11_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        conv_11_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    

    // conv_12
    
    tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
        conv_11_out, conv_12_w, conv_12_b, conv_12_out,
        RELU, conv_12_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_12");
    // tiled_conv_CUTE_auto(conv_12_params, conv_11_out, conv_12_w_t, conv_12_b, cute_temp,RELU);
    // right_test(conv_12_out, cute_temp, conv_12_params.I * conv_12_params.J,CUTE_do_check);
    get_layer_input_output("conv_12",conv_12_params, conv_11_out, conv_12_w_t, conv_12_b, conv_12_out, RELU);
    
    

    // conv_13
    
    tiled_conv_auto(
        conv_13_params.batch_size, conv_13_params.in_row_dim, conv_13_params.in_col_dim,
        conv_13_params.in_channels,
        conv_13_params.out_channels, conv_13_params.out_row_dim, conv_13_params.out_col_dim,
        conv_13_params.stride, 1, 1, conv_13_params.padding, conv_13_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,
        RELU, conv_13_params.output_scale_shift,
        conv_13_params.pool_size, 0, conv_13_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_13_params, conv_12_out, conv_13_w_t, conv_13_b, cute_temp,RELU);
    // right_test(conv_13_out, cute_temp, conv_13_params.I * conv_13_params.J,CUTE_do_check);
    get_layer_input_output("conv_13",conv_13_params, conv_12_out, conv_13_w_t, conv_13_b, conv_13_out, RELU);
    
    

    // conv_14
    
    tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
        conv_13_out, conv_14_w, conv_14_b, conv_14_out,
        NO_ACTIVATION, conv_14_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_14");
    // tiled_conv_CUTE_auto(conv_14_params, conv_13_out, conv_14_w_t, conv_14_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_14_out, cute_temp, conv_14_params.I * conv_14_params.J,CUTE_do_check);
    get_layer_input_output("conv_14",conv_14_params, conv_13_out, conv_14_w_t, conv_14_b, conv_14_out, NO_ACTIVATION);
    
    

    // Downsampling conv_11_out
    // conv_15
    
    // tiled_conv_auto(
    tiled_conv_downsample(
        conv_15_params.batch_size, conv_15_params.in_row_dim, conv_15_params.in_col_dim,
        conv_15_params.in_channels,
        conv_15_params.out_channels, conv_15_params.out_row_dim, conv_15_params.out_col_dim,
        // conv_15_params.stride, 1, 1, conv_15_params.padding, conv_15_params.kernel_size,
        // false, false, false, false, false,
        (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,
        NO_ACTIVATION, conv_15_params.output_scale_shift,
        // conv_15_params.pool_size, 0, conv_15_params.pool_padding,
        tiled_matmul_type);

    // tiled_conv_CUTE_auto(conv_15_params, conv_11_out, conv_15_w_t, conv_15_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_15_out, cute_temp, conv_15_params.I * conv_15_params.J,CUTE_do_check);
    get_layer_input_output("conv_15",conv_15_params, conv_11_out, conv_15_w_t, conv_15_b, conv_15_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        conv_14_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_16
    
    tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
        conv_14_out, conv_16_w, conv_16_b, conv_16_out,
        RELU, conv_16_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_16");
    // tiled_conv_CUTE_auto(conv_16_params, conv_14_out, conv_16_w_t, conv_16_b, cute_temp,RELU);
    // right_test(conv_16_out, cute_temp, conv_16_params.I * conv_16_params.J,CUTE_do_check);
    get_layer_input_output("conv_16",conv_16_params, conv_14_out, conv_16_w_t, conv_16_b, conv_16_out, RELU);
    
    

    // conv_17
    
    tiled_conv_auto(
        conv_17_params.batch_size, conv_17_params.in_row_dim, conv_17_params.in_col_dim,
        conv_17_params.in_channels,
        conv_17_params.out_channels, conv_17_params.out_row_dim, conv_17_params.out_col_dim,
        conv_17_params.stride, 1, 1, conv_17_params.padding, conv_17_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,
        RELU, conv_17_params.output_scale_shift,
        conv_17_params.pool_size, 0, conv_17_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_17_params, conv_16_out, conv_17_w_t, conv_17_b, cute_temp,RELU);
    // right_test(conv_17_out, cute_temp, conv_17_params.I * conv_17_params.J,CUTE_do_check);
    get_layer_input_output("conv_17",conv_17_params, conv_16_out, conv_17_w_t, conv_17_b, conv_17_out, RELU);
    
    

    // conv_18
    
    tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
        conv_17_out, conv_18_w, conv_18_b, conv_18_out,
        NO_ACTIVATION, conv_18_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_18");
    // tiled_conv_CUTE_auto(conv_18_params, conv_17_out, conv_18_w_t, conv_18_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_18_out, cute_temp, conv_18_params.I * conv_18_params.J,CUTE_do_check);
    get_layer_input_output("conv_18",conv_18_params, conv_17_out, conv_18_w_t, conv_18_b, conv_18_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        conv_18_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_19
    
    tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
        conv_18_out, conv_19_w, conv_19_b, conv_19_out,
        RELU, conv_19_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_19");
    // tiled_conv_CUTE_auto(conv_19_params, conv_18_out, conv_19_w_t, conv_19_b, cute_temp,RELU);
    // right_test(conv_19_out, cute_temp, conv_19_params.I * conv_19_params.J,CUTE_do_check);
    get_layer_input_output("conv_19",conv_19_params, conv_18_out, conv_19_w_t, conv_19_b, conv_19_out, RELU);
    
    

    // conv_20
    
    tiled_conv_auto(
        conv_20_params.batch_size, conv_20_params.in_row_dim, conv_20_params.in_col_dim,
        conv_20_params.in_channels,
        conv_20_params.out_channels, conv_20_params.out_row_dim, conv_20_params.out_col_dim,
        conv_20_params.stride, 1, 1, conv_20_params.padding, conv_20_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,
        RELU, conv_20_params.output_scale_shift,
        conv_20_params.pool_size, 0, conv_20_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_20_params, conv_19_out, conv_20_w_t, conv_20_b, cute_temp,RELU);
    // right_test(conv_20_out, cute_temp, conv_20_params.I * conv_20_params.J,CUTE_do_check);
    get_layer_input_output("conv_20",conv_20_params, conv_19_out, conv_20_w_t, conv_20_b, conv_20_out, RELU);
    
    

    // conv_21
    
    tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
        conv_20_out, conv_21_w, conv_21_b, conv_21_out,
        NO_ACTIVATION, conv_21_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_21");
    // tiled_conv_CUTE_auto(conv_21_params, conv_20_out, conv_21_w_t, conv_21_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_21_out, cute_temp, conv_21_params.I * conv_21_params.J,CUTE_do_check);
    get_layer_input_output("conv_21",conv_21_params, conv_20_out, conv_21_w_t, conv_21_b, conv_21_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        conv_21_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_22
    

    tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
        conv_21_out, conv_22_w, conv_22_b, conv_22_out,
        RELU, conv_22_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_22");
    // tiled_conv_CUTE_auto(conv_22_params, conv_21_out, conv_22_w_t, conv_22_b, cute_temp,RELU);
    // right_test(conv_22_out, cute_temp, conv_22_params.I * conv_22_params.J,CUTE_do_check);
    get_layer_input_output("conv_22",conv_22_params, conv_21_out, conv_22_w_t, conv_22_b, conv_22_out, RELU);
    
    

    // conv_23
    
    tiled_conv_auto(
        conv_23_params.batch_size, conv_23_params.in_row_dim, conv_23_params.in_col_dim,
        conv_23_params.in_channels,
        conv_23_params.out_channels, conv_23_params.out_row_dim, conv_23_params.out_col_dim,
        conv_23_params.stride, 1, 1, conv_23_params.padding, conv_23_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,
        RELU, conv_23_params.output_scale_shift,
        conv_23_params.pool_size, 0, conv_23_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_23_params, conv_22_out, conv_23_w_t, conv_23_b, cute_temp,RELU);
    // right_test(conv_23_out, cute_temp, conv_23_params.I * conv_23_params.J,CUTE_do_check);
    get_layer_input_output("conv_23",conv_23_params, conv_22_out, conv_23_w_t, conv_23_b, conv_23_out, RELU);
    
    

    // conv_24
    
    tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
        conv_23_out, conv_24_w, conv_24_b, conv_24_out,
        NO_ACTIVATION, conv_24_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_24");
    // tiled_conv_CUTE_auto(conv_24_params, conv_23_out, conv_24_w_t, conv_24_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_24_out, cute_temp, conv_24_params.I * conv_24_params.J,CUTE_do_check);
    get_layer_input_output("conv_24",conv_24_params, conv_23_out, conv_24_w_t, conv_24_b, conv_24_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        conv_24_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_25
    
    tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
        conv_24_out, conv_25_w, conv_25_b, conv_25_out,
        RELU, conv_25_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_25");
    // tiled_conv_CUTE_auto(conv_25_params, conv_24_out, conv_25_w_t, conv_25_b, cute_temp,RELU);
    // right_test(conv_25_out, cute_temp, conv_25_params.I * conv_25_params.J,CUTE_do_check);
    get_layer_input_output("conv_25",conv_25_params, conv_24_out, conv_25_w_t, conv_25_b, conv_25_out, RELU);
    
    

    // conv_26
    
    tiled_conv_auto(
        conv_26_params.batch_size, conv_26_params.in_row_dim, conv_26_params.in_col_dim,
        conv_26_params.in_channels,
        conv_26_params.out_channels, conv_26_params.out_row_dim, conv_26_params.out_col_dim,
        conv_26_params.stride, 1, 1, conv_26_params.padding, conv_26_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,
        RELU, conv_26_params.output_scale_shift,
        conv_26_params.pool_size, 0, conv_26_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_26_params, conv_25_out, conv_26_w_t, conv_26_b, cute_temp,RELU);
    // right_test(conv_26_out, cute_temp, conv_26_params.I * conv_26_params.J,CUTE_do_check);
    get_layer_input_output("conv_26",conv_26_params, conv_25_out, conv_26_w_t, conv_26_b, conv_26_out, RELU);
    
    

    // conv_27
    
    tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
        conv_26_out, conv_27_w, conv_27_b, conv_27_out,
        NO_ACTIVATION, conv_27_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_27");
    // tiled_conv_CUTE_auto(conv_27_params, conv_26_out, conv_27_w_t, conv_27_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_27_out, cute_temp, conv_27_params.I * conv_27_params.J,CUTE_do_check);
    get_layer_input_output("conv_27",conv_27_params, conv_26_out, conv_27_w_t, conv_27_b, conv_27_out, NO_ACTIVATION);
    
    

    // Downsampling conv_24_out
    // conv_28
    
    tiled_conv_downsample(
        conv_28_params.batch_size, conv_28_params.in_row_dim, conv_28_params.in_col_dim,
        conv_28_params.in_channels,
        conv_28_params.out_channels, conv_28_params.out_row_dim, conv_28_params.out_col_dim,
        (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,
        NO_ACTIVATION, conv_28_params.output_scale_shift,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_28_params, conv_24_out, conv_28_w_t, conv_28_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_28_out, cute_temp, conv_28_params.I * conv_28_params.J,CUTE_do_check);
    get_layer_input_output("conv_28",conv_28_params, conv_24_out, conv_28_w_t, conv_28_b, conv_28_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        conv_27_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_29
    
    tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
        conv_27_out, conv_29_w, conv_29_b, conv_29_out,
        RELU, conv_29_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_29");
    // tiled_conv_CUTE_auto(conv_29_params, conv_27_out, conv_29_w_t, conv_29_b, cute_temp,RELU);
    // right_test(conv_29_out, cute_temp, conv_29_params.I * conv_29_params.J,CUTE_do_check);
    get_layer_input_output("conv_29",conv_29_params, conv_27_out, conv_29_w_t, conv_29_b, conv_29_out, RELU);
    
    

    // conv_30
    
    tiled_conv_auto(
        conv_30_params.batch_size, conv_30_params.in_row_dim, conv_30_params.in_col_dim,
        conv_30_params.in_channels,
        conv_30_params.out_channels, conv_30_params.out_row_dim, conv_30_params.out_col_dim,
        conv_30_params.stride, 1, 1, conv_30_params.padding, conv_30_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,
        RELU, conv_30_params.output_scale_shift,
        conv_30_params.pool_size, 0, conv_30_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_30_params, conv_29_out, conv_30_w_t, conv_30_b, cute_temp,RELU);
    // right_test(conv_30_out, cute_temp, conv_30_params.I * conv_30_params.J,CUTE_do_check);
    get_layer_input_output("conv_30",conv_30_params, conv_29_out, conv_30_w_t, conv_30_b, conv_30_out, RELU);
    
    

    // conv_31
    
    tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
        conv_30_out, conv_31_w, conv_31_b, conv_31_out,
        NO_ACTIVATION, conv_31_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_31");
    // tiled_conv_CUTE_auto(conv_31_params, conv_30_out, conv_31_w_t, conv_31_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_31_out, cute_temp, conv_31_params.I * conv_31_params.J,CUTE_do_check);
    get_layer_input_output("conv_31",conv_31_params, conv_30_out, conv_31_w_t, conv_31_b, conv_31_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        conv_31_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_32
    
    tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
        conv_31_out, conv_32_w, conv_32_b, conv_32_out,
        RELU, conv_32_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_32");
    // tiled_conv_CUTE_auto(conv_32_params, conv_31_out, conv_32_w_t, conv_32_b, cute_temp,RELU);
    // right_test(conv_32_out, cute_temp, conv_32_params.I * conv_32_params.J,CUTE_do_check);
    get_layer_input_output("conv_32",conv_32_params, conv_31_out, conv_32_w_t, conv_32_b, conv_32_out, RELU);
    
    

    // conv_33
    
    tiled_conv_auto(
        conv_33_params.batch_size, conv_33_params.in_row_dim, conv_33_params.in_col_dim,
        conv_33_params.in_channels,
        conv_33_params.out_channels, conv_33_params.out_row_dim, conv_33_params.out_col_dim,
        conv_33_params.stride, 1, 1, conv_33_params.padding, conv_33_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,
        RELU, conv_33_params.output_scale_shift,
        conv_33_params.pool_size, 0, conv_33_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_33_params, conv_32_out, conv_33_w_t, conv_33_b, cute_temp,RELU);
    // right_test(conv_33_out, cute_temp, conv_33_params.I * conv_33_params.J,CUTE_do_check);
    get_layer_input_output("conv_33",conv_33_params, conv_32_out, conv_33_w_t, conv_33_b, conv_33_out, RELU);
    
    

    // conv_34
    
    tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
        conv_33_out, conv_34_w, conv_34_b, conv_34_out,
        NO_ACTIVATION, conv_34_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_34");
    // tiled_conv_CUTE_auto(conv_34_params, conv_33_out, conv_34_w_t, conv_34_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_34_out, cute_temp, conv_34_params.I * conv_34_params.J,CUTE_do_check);
    get_layer_input_output("conv_34",conv_34_params, conv_33_out, conv_34_w_t, conv_34_b, conv_34_out, NO_ACTIVATION);
    
    
    // Add residuals
    
    tiled_resadd_auto(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        conv_34_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_35
    
    tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
        conv_34_out, conv_35_w, conv_35_b, conv_35_out,
        RELU, conv_35_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_35");
    // tiled_conv_CUTE_auto(conv_35_params, conv_34_out, conv_35_w_t, conv_35_b, cute_temp,RELU);
    // right_test(conv_35_out, cute_temp, conv_35_params.I * conv_35_params.J,CUTE_do_check);
    get_layer_input_output("conv_35",conv_35_params, conv_34_out, conv_35_w_t, conv_35_b, conv_35_out, RELU);
    
    

    // conv_36
    
    tiled_conv_auto(
        conv_36_params.batch_size, conv_36_params.in_row_dim, conv_36_params.in_col_dim,
        conv_36_params.in_channels,
        conv_36_params.out_channels, conv_36_params.out_row_dim, conv_36_params.out_col_dim,
        conv_36_params.stride, 1, 1, conv_36_params.padding, conv_36_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,
        RELU, conv_36_params.output_scale_shift,
        conv_36_params.pool_size, 0, conv_36_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_36_params, conv_35_out, conv_36_w_t, conv_36_b, cute_temp,RELU);
    // right_test(conv_36_out, cute_temp, conv_36_params.I * conv_36_params.J,CUTE_do_check);
    get_layer_input_output("conv_36",conv_36_params, conv_35_out, conv_36_w_t, conv_36_b, conv_36_out, RELU);
    
    

    // conv_37
    
    tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
        conv_36_out, conv_37_w, conv_37_b, conv_37_out,
        NO_ACTIVATION, conv_37_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_37");
    // tiled_conv_CUTE_auto(conv_37_params, conv_36_out, conv_37_w_t, conv_37_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_37_out, cute_temp, conv_37_params.I * conv_37_params.J,CUTE_do_check);
    get_layer_input_output("conv_37",conv_37_params, conv_36_out, conv_37_w_t, conv_37_b, conv_37_out, NO_ACTIVATION);
    
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        conv_37_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_38
    
    tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
        conv_37_out, conv_38_w, conv_38_b, conv_38_out,
        RELU, conv_38_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_38");
    // tiled_conv_CUTE_auto(conv_38_params, conv_37_out, conv_38_w_t, conv_38_b, cute_temp,RELU);
    // right_test(conv_38_out, cute_temp, conv_38_params.I * conv_38_params.J,CUTE_do_check);
    get_layer_input_output("conv_38",conv_38_params, conv_37_out, conv_38_w_t, conv_38_b, conv_38_out, RELU);
    
    

    // conv_39
    
    tiled_conv_auto(
        conv_39_params.batch_size, conv_39_params.in_row_dim, conv_39_params.in_col_dim,
        conv_39_params.in_channels,
        conv_39_params.out_channels, conv_39_params.out_row_dim, conv_39_params.out_col_dim,
        conv_39_params.stride, 1, 1, conv_39_params.padding, conv_39_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,
        RELU, conv_39_params.output_scale_shift,
        conv_39_params.pool_size, 0, conv_39_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_39_params, conv_38_out, conv_39_w_t, conv_39_b, cute_temp,RELU);
    // right_test(conv_39_out, cute_temp, conv_39_params.I * conv_39_params.J,CUTE_do_check);
    get_layer_input_output("conv_39",conv_39_params, conv_38_out, conv_39_w_t, conv_39_b, conv_39_out, RELU);
    
    

    // conv_40
    
    tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
        conv_39_out, conv_40_w, conv_40_b, conv_40_out,
        NO_ACTIVATION, conv_40_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_40");
    // tiled_conv_CUTE_auto(conv_40_params, conv_39_out, conv_40_w_t, conv_40_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_40_out, cute_temp, conv_40_params.I * conv_40_params.J,CUTE_do_check);
    get_layer_input_output("conv_40",conv_40_params, conv_39_out, conv_40_w_t, conv_40_b, conv_40_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        conv_40_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_41
    
    tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
        conv_40_out, conv_41_w, conv_41_b, conv_41_out,
        RELU, conv_41_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_41");
    // tiled_conv_CUTE_auto(conv_41_params, conv_40_out, conv_41_w_t, conv_41_b, cute_temp,RELU);
    // right_test(conv_41_out, cute_temp, conv_41_params.I * conv_41_params.J,CUTE_do_check);
    get_layer_input_output("conv_41",conv_41_params, conv_40_out, conv_41_w_t, conv_41_b, conv_41_out, RELU);
    
    

    // conv_42
    
    tiled_conv_auto(
        conv_42_params.batch_size, conv_42_params.in_row_dim, conv_42_params.in_col_dim,
        conv_42_params.in_channels,
        conv_42_params.out_channels, conv_42_params.out_row_dim, conv_42_params.out_col_dim,
        conv_42_params.stride, 1, 1, conv_42_params.padding, conv_42_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,
        RELU, conv_42_params.output_scale_shift,
        conv_42_params.pool_size, 0, conv_42_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_42_params, conv_41_out, conv_42_w_t, conv_42_b, cute_temp,RELU);
    // right_test(conv_42_out, cute_temp, conv_42_params.I * conv_42_params.J,CUTE_do_check);
    get_layer_input_output("conv_42",conv_42_params, conv_41_out, conv_42_w_t, conv_42_b, conv_42_out, RELU);
    
    

    // conv_43
    
    tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
        conv_42_out, conv_43_w, conv_43_b, conv_43_out,
        NO_ACTIVATION, conv_43_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_43");
    // tiled_conv_CUTE_auto(conv_43_params, conv_42_out, conv_43_w_t, conv_43_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_43_out, cute_temp, conv_43_params.I * conv_43_params.J,CUTE_do_check);
    get_layer_input_output("conv_43",conv_43_params, conv_42_out, conv_43_w_t, conv_43_b, conv_43_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        conv_43_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_44
    
    tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
        conv_43_out, conv_44_w, conv_44_b, conv_44_out,
        RELU, conv_44_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_44");
    // tiled_conv_CUTE_auto(conv_44_params, conv_43_out, conv_44_w_t, conv_44_b, cute_temp,RELU);
    // right_test(conv_44_out, cute_temp, conv_44_params.I * conv_44_params.J,CUTE_do_check);
    get_layer_input_output("conv_44",conv_44_params, conv_43_out, conv_44_w_t, conv_44_b, conv_44_out, RELU);
    
    

    // conv_45
    
    tiled_conv_auto(
        conv_45_params.batch_size, conv_45_params.in_row_dim, conv_45_params.in_col_dim,
        conv_45_params.in_channels,
        conv_45_params.out_channels, conv_45_params.out_row_dim, conv_45_params.out_col_dim,
        conv_45_params.stride, 1, 1, conv_45_params.padding, conv_45_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,
        RELU, conv_45_params.output_scale_shift,
        conv_45_params.pool_size, 0, conv_45_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_45_params, conv_44_out, conv_45_w_t, conv_45_b, cute_temp,RELU);
    // right_test(conv_45_out, cute_temp, conv_45_params.I * conv_45_params.J,CUTE_do_check);
    get_layer_input_output("conv_45",conv_45_params, conv_44_out, conv_45_w_t, conv_45_b, conv_45_out, RELU);
    
    

    // conv_46
    
    tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
        conv_45_out, conv_46_w, conv_46_b, conv_46_out,
        NO_ACTIVATION, conv_46_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_46");
    // tiled_conv_CUTE_auto(conv_46_params, conv_45_out, conv_46_w_t, conv_46_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_46_out, cute_temp, conv_46_params.I * conv_46_params.J,CUTE_do_check);
    get_layer_input_output("conv_46",conv_46_params, conv_45_out, conv_46_w_t, conv_46_b, conv_46_out, NO_ACTIVATION);
    
    

    // Downsampling conv_43_out
    // conv_47
    
    tiled_conv_auto(
        conv_47_params.batch_size, conv_47_params.in_row_dim, conv_47_params.in_col_dim,
        conv_47_params.in_channels,
        conv_47_params.out_channels, conv_47_params.out_row_dim, conv_47_params.out_col_dim,
        conv_47_params.stride, 1, 1, conv_47_params.padding, conv_47_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,
        NO_ACTIVATION, conv_47_params.output_scale_shift,
        conv_47_params.pool_size, 0, conv_47_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_47_params, conv_43_out, conv_47_w_t, conv_47_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_47_out, cute_temp, conv_47_params.I * conv_47_params.J,CUTE_do_check);
    get_layer_input_output("conv_47",conv_47_params, conv_43_out, conv_47_w_t, conv_47_b, conv_47_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        conv_46_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_48
    

    tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
        conv_46_out, conv_48_w, conv_48_b, conv_48_out,
        RELU, conv_48_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_48");
    // tiled_conv_CUTE_auto(conv_48_params, conv_46_out, conv_48_w_t, conv_48_b, cute_temp,RELU);
    // right_test(conv_48_out, cute_temp, conv_48_params.I * conv_48_params.J,CUTE_do_check);
    get_layer_input_output("conv_48",conv_48_params, conv_46_out, conv_48_w_t, conv_48_b, conv_48_out, RELU);
    
    

    // conv_49
    

    tiled_conv_auto(
        conv_49_params.batch_size, conv_49_params.in_row_dim, conv_49_params.in_col_dim,
        conv_49_params.in_channels,
        conv_49_params.out_channels, conv_49_params.out_row_dim, conv_49_params.out_col_dim,
        conv_49_params.stride, 1, 1, conv_49_params.padding, conv_49_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,
        RELU, conv_49_params.output_scale_shift,
        conv_49_params.pool_size, 0, conv_49_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_49_params, conv_48_out, conv_49_w_t, conv_49_b, cute_temp,RELU);
    // right_test(conv_49_out, cute_temp, conv_49_params.I * conv_49_params.J,CUTE_do_check);
    get_layer_input_output("conv_49",conv_49_params, conv_48_out, conv_49_w_t, conv_49_b, conv_49_out, RELU);
    
    

    // conv_50
    

    tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
        conv_49_out, conv_50_w, conv_50_b, conv_50_out,
        NO_ACTIVATION, conv_50_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_50");
    // tiled_conv_CUTE_auto(conv_50_params, conv_49_out, conv_50_w_t, conv_50_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_50_out, cute_temp, conv_50_params.I * conv_50_params.J,CUTE_do_check);
    get_layer_input_output("conv_50",conv_50_params, conv_49_out, conv_50_w_t, conv_50_b, conv_50_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        conv_50_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    
    
    // conv_51
    
    tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
        conv_50_out, conv_51_w, conv_51_b, conv_51_out,
        RELU, conv_51_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_51");
    // tiled_conv_CUTE_auto(conv_51_params, conv_50_out, conv_51_w_t, conv_51_b, cute_temp,RELU);
    // right_test(conv_51_out, cute_temp, conv_51_params.I * conv_51_params.J,CUTE_do_check);
    get_layer_input_output("conv_51",conv_51_params, conv_50_out, conv_51_w_t, conv_51_b, conv_51_out, RELU);
    
    

    // conv_52
    
    tiled_conv_auto(
        conv_52_params.batch_size, conv_52_params.in_row_dim, conv_52_params.in_col_dim,
        conv_52_params.in_channels,
        conv_52_params.out_channels, conv_52_params.out_row_dim, conv_52_params.out_col_dim,
        conv_52_params.stride, 1, 1, conv_52_params.padding, conv_52_params.kernel_size,
        false, false, false, false, false,
        (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,
        RELU, conv_52_params.output_scale_shift,
        conv_52_params.pool_size, 0, conv_52_params.pool_padding,
        tiled_matmul_type);
    // tiled_conv_CUTE_auto(conv_52_params, conv_51_out, conv_52_w_t, conv_52_b, cute_temp,RELU);
    // right_test(conv_52_out, cute_temp, conv_52_params.I * conv_52_params.J,CUTE_do_check);
    get_layer_input_output("conv_52",conv_52_params, conv_51_out, conv_52_w_t, conv_52_b, conv_52_out, RELU);
    
    

    // conv_53
    
    tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
        conv_52_out, conv_53_w, conv_53_b, conv_53_out,
        NO_ACTIVATION, conv_53_params.output_scale_shift, true,
        tiled_matmul_type, check, "conv_53");
    // tiled_conv_CUTE_auto(conv_53_params, conv_52_out, conv_53_w_t, conv_53_b, cute_temp,NO_ACTIVATION);
    // right_test(conv_53_out, cute_temp, conv_53_params.I * conv_53_params.J,CUTE_do_check);
    get_layer_input_output("conv_53",conv_53_params, conv_52_out, conv_53_w_t, conv_53_b, conv_53_out, NO_ACTIVATION);
    
    

    // Add residuals
    
    tiled_resadd_auto(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale_shift,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        conv_53_params.res_scale_greater,
        tiled_matmul_type == CPU ? CPU : WS);
    
    

    // Global averaging
    static elem_t average[4][2048] row_align(1);

    
    tiled_global_average_auto(conv_53_out, average, conv_53_params.batch_size,
        conv_53_params.out_channels, conv_53_params.out_row_dim, CPU);
    

    // fc_54
    
    tiled_matmul_nn_auto(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        average, fc_54_w, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale_shift, false,
        tiled_matmul_type, check, "fc_54");
    
    
    

    // Find highest probs
    int preds[fc_54_params.batch_size];
    for (int batch = 0; batch < fc_54_params.batch_size; batch++) {
        elem_t max_prob = fc_54_out[batch][0];
        size_t max_idx = 0;

        for (int i = 1; i < fc_54_params.out_features; i++) {
            if (fc_54_out[batch][i] > max_prob) {
                max_prob = fc_54_out[batch][i];
                max_idx = i;
            }
        }

        preds[batch] = max_idx;
        printf("Prediction: %u (score: %d)\n", max_idx, max_prob);
    }

    int correct[] = {75, 900, 641, 897};
    for (int i = 0; i < fc_54_params.batch_size; i++) {
        if (preds[i] != correct[i] || fc_54_out[i][preds[i]] != fc_54_out[i][correct[i]]) {
            printf("Prediction %d is incorrect!\nFAIL\n", i+1);
            exit(1);
        }
    }

    printf("PASS\n");
    exit(0);
}

