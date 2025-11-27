#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#include "resnet50_params_gloden.h"
#include "images.h"
#include <stdlib.h>

#define GEMMINI_ACC_SCALE(x, scale) (x)
#define GEMMINI_SCALE(x, scale) (x)

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}
static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

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

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = sum / count;
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
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

    if (matadd_type == CPU) {
        resadd_cpu(I, J,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
                //输出前1000个输出
    printf("tiled_resadd_auto:");
    for (int i = 0; i < 1000; i++) {
        printf("%d ",*(C+i));
    }
    printf("\n");
        return;
    }


}


static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

  const int no_bias = D == NULL;
  //输出所有输入参数
    // printf("transA:%d,transB:%d,DIM_I:%d,DIM_J:%d,DIM_K:%d,stride_A:%d,stride_B:%d,stride_D:%d,stride_C:%d\n A_scale_factor:%d,B_scale_factor:%d,D_scale_factor:%d,act:%d,scale:%d,bert_scale:%d,repeating_bias:%d\n",transA,transB,DIM_I,DIM_J,DIM_K,stride_A,stride_B,stride_D,stride_C,A_scale_factor,B_scale_factor,D_scale_factor,act,scale,bert_scale,repeating_bias);
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
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {



    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
  
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
        int act, acc_scale_t scale, bool repeating_bias,
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
        int act, acc_scale_t scale, bool repeating_bias,
        enum tiled_matmul_type_t tiled_matmul_type,
        bool check, char * layer_name)
{
    if (check)
        printf("%s: gemmini\n", layer_name);

    tiled_matmul_auto(dim_I, dim_J, dim_K,
        (elem_t*)A, (elem_t*)B, D, (elem_t*)C, 
        dim_K, dim_J, dim_J, dim_J,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        act, scale, 0, repeating_bias,
        false, false,
        false, false,
        0,
        tiled_matmul_type);

    //输出前1000个元素
    printf("tiled_matmul_nn_auto:");
    for(int i=0;i<1000;i++){
        printf("%d ",*((elem_t*)C+i));
    }
    printf("\n");
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

static void tiled_conv_downsample(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,

        enum tiled_matmul_type_t tiled_conv_type) {

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
}

static void conv_dw(size_t I, size_t J,
    const size_t batch_size, const size_t channels,
    const size_t in_row_dim, const size_t in_col_dim,
    const size_t out_row_dim, const size_t out_col_dim,
    const size_t kernel_size,
    const elem_t input[batch_size][in_row_dim][in_col_dim][channels],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_row_dim][out_col_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_row_dim; out_row++) {
                for (int out_col = 0; out_col < out_col_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_row_dim && in_col >= 0 && in_col < params->in_col_dim) {
                                result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_row_dim * params->out_col_dim + out_row * params->out_col_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
}

static void conv_dw_with_col2im(size_t prev_I, size_t prev_J, size_t I, size_t J,
    const size_t batch_size, const size_t channels,
    const size_t out_row_dim, const size_t out_col_dim, const size_t kernel_size,
    const elem_t input[prev_I][prev_J],
    const elem_t weight[channels][kernel_size][kernel_size],
    const acc_t * bias,
    // elem_t output [batch_size][out_dim][out_dim][channels],
    elem_t output [I][J],
    const struct ConvParams * params)
{
    for (int batch = 0; batch < batch_size; batch++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int out_row = 0; out_row < out_row_dim; out_row++) {
                for (int out_col = 0; out_col < out_col_dim; out_col++) {
                    int in_row = out_row * params->stride - params->padding;

                    acc_t result = 0;
                    if (params->bias) {
                        result = bias[channel];
                    }

                    for (int kernel_row = 0; kernel_row < params->kernel_size; kernel_row++) {
                        int in_col = out_col * params->stride - params->padding;

                        for (int kernel_col = 0; kernel_col < params->kernel_size; kernel_col++) {
                            if (in_row >= 0 && in_row < params->in_row_dim && in_col >= 0 && in_col < params->in_col_dim) {
                                // result += input[batch][in_row][in_col][channel] * weight[channel][kernel_row][kernel_col];

                                size_t r = batch * params->in_row_dim * params->in_col_dim + in_row * params->in_col_dim + in_col;

                                result += input[r][channel] * weight[channel][kernel_row][kernel_col];
                            }

                            in_col++;
                        }

                        in_row++;
                    }

                    if (result < 0) {
                        result = 0;
                    }
                    
                    acc_t scaled = ACC_SCALE(result, params->output_scale);

                    if (scaled > elem_t_max) {
                        scaled = elem_t_max;
                    } else if (scaled < elem_t_min) {
                        scaled = elem_t_min;
                    }
                    
                    size_t r = batch * params->out_row_dim * params->out_col_dim + out_row * params->out_col_dim + out_col;
                    output[r][channel] = scaled;
                    // output[batch][out_row][out_col][channel] = scaled;
                }
            }
        }
    }
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

// Compute C = A + B with saturating add
void vecadd(size_t len, const elem_t * A, const elem_t * B, elem_t * C, scale_t A_shift) {
    for (size_t i = 0; i < len; i++) {
        acc_t result = MVIN_SCALE(A[i], A_shift) + B[i];

        if (result > elem_t_max) {
            result = elem_t_max;
        } else if (result < elem_t_min) {
            result = elem_t_min;
        }

        C[i] = result;
    }
}

void resadd1(const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[batch_size][im_dim][im_dim][channels],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    acc_t result = MVIN_SCALE(A[batch][row][col][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd2(const size_t I, const size_t J,
    const size_t batch_size, const size_t channels, const size_t im_dim,
    const elem_t A[I][J],
    const elem_t B[batch_size][im_dim][im_dim][channels],
    elem_t C[batch_size][im_dim][im_dim][channels],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[batch][row][col][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[batch][row][col][channel] = result;
                }
            }
        }
    }
}

void resadd3(const size_t I, const size_t J,
    const elem_t A[I][J],
    const elem_t B[I][J],
    elem_t C[I][J],
    bool relu,
    const struct ConvParams * params) {

    const int minimum = relu ? 0 : elem_t_min;

    for (size_t batch = 0; batch < params->batch_size; batch++) {
        for (size_t row = 0; row < params->out_dim_pooled; row++) {
            for (size_t col = 0; col < params->out_dim_pooled; col++) {
                for (size_t channel = 0; channel < params->out_channels; channel++) {
                    size_t r = batch * params->out_dim_pooled * params->out_dim_pooled + row * params->out_dim_pooled + col;

                    acc_t result = MVIN_SCALE(A[r][channel], params->res_scale) + B[r][channel];

                    if (result > elem_t_max) {
                        result = elem_t_max;
                    } else if (result < minimum) {
                        result = minimum;
                    }

                    C[r][channel] = result;
                }
            }
        }
    }
}

// Pooling
void pool(size_t batch_size, size_t channels, size_t in_row_dim, size_t in_col_dim,
    size_t out_row_dim, size_t out_col_dim,
    elem_t input[batch_size][in_row_dim][in_col_dim][channels],
    elem_t output[batch_size][out_row_dim][out_col_dim][channels],
    const struct ConvParams * params)
{
    size_t kernel_size = params->pool_size;
    size_t stride = params->pool_stride;
    // size_t in_dim = params->out_dim;
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
                                if (input[batch][in_row][in_col][channel] > result) {
                                    result = input[batch][in_row][in_col][channel];
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

        int act, acc_scale_t scale) {

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

          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
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

        int act, acc_scale_t scale,
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

        int act, acc_scale_t scale,
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

        int act, acc_scale_t scale,
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

    //输出前1000项的值
    printf("conv_auto: ");
    for (int i = 0; i < 1000; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");
}


int main (int argc, char * argv[]) {

    // gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type = WS;

    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool conv = true;
    
    if (argc < 3) {
        conv = true;
    } else if (strcmp(argv[2], "conv") == 0) {
        conv = true;
    } else if (strcmp(argv[2], "matmul") == 0) {
        conv = false;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check] [conv]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check = false;

    if (argc < 4) {
        check = false;
    } else if (strcmp(argv[3], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, conv_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // conv_1
    if (!conv) {
      start = read_cycles();

        im2col(conv_1_params.batch_size, conv_1_params.in_channels,
            conv_1_params.in_row_dim, conv_1_params.in_col_dim,
            conv_1_params.I, conv_1_params.K,
            images, conv_1_in, &conv_1_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
            conv_1_in, conv_1_w, conv_1_b, conv_1_out,
            RELU, conv_1_params.output_scale, true,
            tiled_matmul_type, check, "conv_1");

        end = read_cycles();
        matmul_cycles += end - start;

      start = read_cycles();

        pool_with_col2im(conv_1_params.I, conv_1_params.J,
            conv_1_params.batch_size, conv_1_params.out_channels,
            conv_1_params.out_dim_pooled,
            conv_1_params.out_dim_pooled,
            conv_1_out, conv_1_out_pooled, &conv_1_params);

        end = read_cycles();
        pool_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_1_params.batch_size, conv_1_params.in_row_dim, conv_1_params.in_col_dim,
            conv_1_params.in_channels,
            conv_1_params.out_channels, conv_1_params.out_row_dim, conv_1_params.out_col_dim,
            conv_1_params.stride, 1, 1, conv_1_params.padding, conv_1_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)images, (elem_t*)conv_1_w, (acc_t*)conv_1_b, (elem_t*)conv_1_out_pooled,

            RELU, conv_1_params.output_scale,
            conv_1_params.pool_size, conv_1_params.pool_stride, conv_1_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 1 cycles: %llu \n", end - start);
    }

    // conv_2
    if (!conv) {
      start = read_cycles();

        im2col(conv_2_params.batch_size, conv_2_params.in_channels,
            conv_2_params.in_row_dim, conv_2_params.in_col_dim,
            conv_2_params.I, conv_2_params.K,
            conv_1_out_pooled, conv_2_in, &conv_2_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_2_in, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_2_params.I, conv_2_params.J, conv_2_params.K,
            conv_1_out_pooled, conv_2_w, conv_2_b, conv_2_out,
            RELU, conv_2_params.output_scale, true,
            tiled_matmul_type, check, "conv_2");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 2 cycles: %llu \n", end - start);
    }

    // conv_3
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_2_params.I, conv_2_params.J,
            conv_3_params.I, conv_3_params.K,
            conv_2_out, conv_3_in, &conv_3_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_3_params.I, conv_3_params.J, conv_3_params.K,
            conv_3_in, conv_3_w, conv_3_b, conv_3_out,
            RELU, conv_3_params.output_scale, true,
            tiled_matmul_type, check, "conv_3");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_3_params.batch_size, conv_3_params.in_row_dim, conv_3_params.in_col_dim,
            conv_3_params.in_channels,
            conv_3_params.out_channels, conv_3_params.out_row_dim, conv_3_params.out_col_dim,
            conv_3_params.stride, 1, 1, conv_3_params.padding, conv_3_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_2_out, (elem_t*)conv_3_w, (acc_t*)conv_3_b, (elem_t*)conv_3_out,

            RELU, conv_3_params.output_scale,
            conv_3_params.pool_size, 0, conv_3_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 3 cycles: %llu \n", end - start);
    }

    // conv_4
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            NO_ACTIVATION, conv_4_params.output_scale, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_4_params.I, conv_4_params.J, conv_4_params.K,
            conv_3_out, conv_4_w, conv_4_b, conv_4_out,
            NO_ACTIVATION, conv_4_params.output_scale, true,
            tiled_matmul_type, check, "conv_4");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 4 cycles: %llu \n", end - start);
    }

    // Downsampling conv_1_out_pooled
    // conv_5
    if (!conv) {
      start = read_cycles();

        im2col(conv_5_params.batch_size, conv_5_params.in_channels,
            conv_5_params.in_row_dim, conv_5_params.in_col_dim,
            conv_5_params.I, conv_5_params.K,
            conv_1_out_pooled, conv_5_in, &conv_5_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_5_in, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_5_params.I, conv_5_params.J, conv_5_params.K,
            conv_1_out_pooled, conv_5_w, conv_5_b, conv_5_out,
            NO_ACTIVATION, conv_5_params.output_scale, true,
            tiled_matmul_type, check, "conv_5");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 5 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_4_params.I, conv_4_params.J,
        conv_4_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_5_out,
        conv_4_out,
        conv_4_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;

    // conv_6
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_4_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_6_params.I, conv_6_params.J, conv_6_params.K,
            conv_4_out, conv_6_w, conv_6_b, conv_6_out,
            RELU, conv_6_params.output_scale, true,
            tiled_matmul_type, check, "conv_6");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 6 cycles: %llu \n", end - start);
    }

    // conv_7
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_6_params.I, conv_6_params.J,
            conv_7_params.I, conv_7_params.K,
            conv_6_out, conv_7_in, &conv_7_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_7_params.I, conv_7_params.J, conv_7_params.K,
            conv_7_in, conv_7_w, conv_7_b, conv_7_out,
            RELU, conv_7_params.output_scale, true,
            tiled_matmul_type, check, "conv_7");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_7_params.batch_size, conv_7_params.in_row_dim, conv_7_params.in_col_dim,
            conv_7_params.in_channels,
            conv_7_params.out_channels, conv_7_params.out_row_dim, conv_7_params.out_col_dim,
            conv_7_params.stride, 1, 1, conv_7_params.padding, conv_7_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_6_out, (elem_t*)conv_7_w, (acc_t*)conv_7_b, (elem_t*)conv_7_out,

            RELU, conv_7_params.output_scale,
            conv_7_params.pool_size, 0, conv_7_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 7 cycles: %llu \n", end - start);
    }

    // conv_8
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out, conv_8_w, conv_8_b, conv_8_out,
            NO_ACTIVATION, conv_8_params.output_scale, true,
            tiled_matmul_type, check, "conv_8");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_8_params.I, conv_8_params.J, conv_8_params.K,
            conv_7_out, conv_8_w, conv_8_b, conv_8_out,
            NO_ACTIVATION, conv_8_params.output_scale, true,
            tiled_matmul_type, check, "conv_8");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 8 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_8_params.I, conv_8_params.J,
        conv_8_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_4_out,
        conv_8_out,
        conv_8_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;

    // conv_9
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_9_params.I, conv_9_params.J, conv_9_params.K,
            conv_8_out, conv_9_w, conv_9_b, conv_9_out,
            RELU, conv_9_params.output_scale, true,
            tiled_matmul_type, check, "conv_9");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 9 cycles: %llu \n", end - start);
    }

    // conv_10
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_9_params.I, conv_9_params.J,
            conv_10_params.I, conv_10_params.K,
            conv_9_out, conv_10_in, &conv_10_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_10_params.I, conv_10_params.J, conv_10_params.K,
            conv_10_in, conv_10_w, conv_10_b, conv_10_out,
            RELU, conv_10_params.output_scale, true,
            tiled_matmul_type, check, "conv_10");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_10_params.batch_size, conv_10_params.in_row_dim, conv_10_params.in_col_dim,
            conv_10_params.in_channels,
            conv_10_params.out_channels, conv_10_params.out_row_dim, conv_10_params.out_col_dim,
            conv_10_params.stride, 1, 1, conv_10_params.padding, conv_10_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_9_out, (elem_t*)conv_10_w, (acc_t*)conv_10_b, (elem_t*)conv_10_out,

            RELU, conv_10_params.output_scale,
            conv_10_params.pool_size, 0, conv_10_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 10 cycles: %llu \n", end - start);
    }

    // conv_11
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            NO_ACTIVATION, conv_11_params.output_scale, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_11_params.I, conv_11_params.J, conv_11_params.K,
            conv_10_out, conv_11_w, conv_11_b, conv_11_out,
            NO_ACTIVATION, conv_11_params.output_scale, true,
            tiled_matmul_type, check, "conv_11");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 11 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_11_params.I, conv_11_params.J,
        conv_11_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_8_out,
        conv_11_out,
        conv_11_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;

    // conv_12
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_12_params.I, conv_12_params.J, conv_12_params.K,
            conv_11_out, conv_12_w, conv_12_b, conv_12_out,
            RELU, conv_12_params.output_scale, true,
            tiled_matmul_type, check, "conv_12");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 12 cycles: %llu \n", end - start);
    }

    // conv_13
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_12_params.I, conv_12_params.J,
            conv_13_params.I, conv_13_params.K,
            conv_12_out, conv_13_in, &conv_13_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_13_params.I, conv_13_params.J, conv_13_params.K,
            conv_13_in, conv_13_w, conv_13_b, conv_13_out,
            RELU, conv_13_params.output_scale, true,
            tiled_matmul_type, check, "conv_13");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_13_params.batch_size, conv_13_params.in_row_dim, conv_13_params.in_col_dim,
            conv_13_params.in_channels,
            conv_13_params.out_channels, conv_13_params.out_row_dim, conv_13_params.out_col_dim,
            conv_13_params.stride, 1, 1, conv_13_params.padding, conv_13_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_12_out, (elem_t*)conv_13_w, (acc_t*)conv_13_b, (elem_t*)conv_13_out,

            RELU, conv_13_params.output_scale,
            conv_13_params.pool_size, 0, conv_13_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 13 cycles: %llu \n", end - start);
    }

    // conv_14
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out, conv_14_w, conv_14_b, conv_14_out,
            NO_ACTIVATION, conv_14_params.output_scale, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_14_params.I, conv_14_params.J, conv_14_params.K,
            conv_13_out, conv_14_w, conv_14_b, conv_14_out,
            NO_ACTIVATION, conv_14_params.output_scale, true,
            tiled_matmul_type, check, "conv_14");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 14 cycles: %llu \n", end - start);
    }

    // Downsampling conv_11_out
    // conv_15
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_11_params.I, conv_11_params.J,
            conv_15_params.I, conv_15_params.K,
            conv_11_out, conv_15_in, &conv_15_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_15_params.I, conv_15_params.J, conv_15_params.K,
            conv_15_in, conv_15_w, conv_15_b, conv_15_out,
            NO_ACTIVATION, conv_15_params.output_scale, true,
            tiled_matmul_type, check, "conv_15");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_15_params.batch_size, conv_15_params.in_row_dim, conv_15_params.in_col_dim,
            conv_15_params.in_channels,
            conv_15_params.out_channels, conv_15_params.out_row_dim, conv_15_params.out_col_dim,
            // conv_15_params.stride, 1, 1, conv_15_params.padding, conv_15_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_11_out, (elem_t*)conv_15_w, (acc_t*)conv_15_b, (elem_t*)conv_15_out,

            NO_ACTIVATION, conv_15_params.output_scale,
            // conv_15_params.pool_size, 0, conv_15_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 15 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_14_params.I, conv_14_params.J,
        conv_14_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_15_out,
        conv_14_out,
        conv_14_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_16
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_14_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_16_params.I, conv_16_params.J, conv_16_params.K,
            conv_14_out, conv_16_w, conv_16_b, conv_16_out,
            RELU, conv_16_params.output_scale, true,
            tiled_matmul_type, check, "conv_16");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 16 cycles: %llu \n", end - start);
    }

    // conv_17
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_16_params.I, conv_16_params.J,
            conv_17_params.I, conv_17_params.K,
            conv_16_out, conv_17_in, &conv_17_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_17_params.I, conv_17_params.J, conv_17_params.K,
            conv_17_in, conv_17_w, conv_17_b, conv_17_out,
            RELU, conv_17_params.output_scale, true,
            tiled_matmul_type, check, "conv_17");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_17_params.batch_size, conv_17_params.in_row_dim, conv_17_params.in_col_dim,
            conv_17_params.in_channels,
            conv_17_params.out_channels, conv_17_params.out_row_dim, conv_17_params.out_col_dim,
            conv_17_params.stride, 1, 1, conv_17_params.padding, conv_17_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_16_out, (elem_t*)conv_17_w, (acc_t*)conv_17_b, (elem_t*)conv_17_out,

            RELU, conv_17_params.output_scale,
            conv_17_params.pool_size, 0, conv_17_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 17 cycles: %llu \n", end - start);
    }

    // conv_18
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_18_params.I, conv_18_params.J, conv_18_params.K,
            conv_17_out, conv_18_w, conv_18_b, conv_18_out,
            NO_ACTIVATION, conv_18_params.output_scale, true,
            tiled_matmul_type, check, "conv_18");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 18 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_18_params.I, conv_18_params.J,
        conv_18_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_14_out,
        conv_18_out,
        conv_18_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_19
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_19_params.I, conv_19_params.J, conv_19_params.K,
            conv_18_out, conv_19_w, conv_19_b, conv_19_out,
            RELU, conv_19_params.output_scale, true,
            tiled_matmul_type, check, "conv_19");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 19 cycles: %llu \n", end - start);
    }

    // conv_20
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_19_params.I, conv_19_params.J,
            conv_20_params.I, conv_20_params.K,
            conv_19_out, conv_20_in, &conv_20_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_20_params.I, conv_20_params.J, conv_20_params.K,
            conv_20_in, conv_20_w, conv_20_b, conv_20_out,
            RELU, conv_20_params.output_scale, true,
            tiled_matmul_type, check, "conv_20");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_20_params.batch_size, conv_20_params.in_row_dim, conv_20_params.in_col_dim,
            conv_20_params.in_channels,
            conv_20_params.out_channels, conv_20_params.out_row_dim, conv_20_params.out_col_dim,
            conv_20_params.stride, 1, 1, conv_20_params.padding, conv_20_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_19_out, (elem_t*)conv_20_w, (acc_t*)conv_20_b, (elem_t*)conv_20_out,

            RELU, conv_20_params.output_scale,
            conv_20_params.pool_size, 0, conv_20_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 20 cycles: %llu \n", end - start);
    }

    // conv_21
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_21_params.I, conv_21_params.J, conv_21_params.K,
            conv_20_out, conv_21_w, conv_21_b, conv_21_out,
            NO_ACTIVATION, conv_21_params.output_scale, true,
            tiled_matmul_type, check, "conv_21");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 21 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_21_params.I, conv_21_params.J,
        conv_21_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_18_out,
        conv_21_out,
        conv_21_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_22
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_22_params.I, conv_22_params.J, conv_22_params.K,
            conv_21_out, conv_22_w, conv_22_b, conv_22_out,
            RELU, conv_22_params.output_scale, true,
            tiled_matmul_type, check, "conv_22");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 22 cycles: %llu \n", end - start);
    }

    // conv_23
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_22_params.I, conv_22_params.J,
            conv_23_params.I, conv_23_params.K,
            conv_22_out, conv_23_in, &conv_23_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_23_params.I, conv_23_params.J, conv_23_params.K,
            conv_23_in, conv_23_w, conv_23_b, conv_23_out,
            RELU, conv_23_params.output_scale, true,
            tiled_matmul_type, check, "conv_23");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_23_params.batch_size, conv_23_params.in_row_dim, conv_23_params.in_col_dim,
            conv_23_params.in_channels,
            conv_23_params.out_channels, conv_23_params.out_row_dim, conv_23_params.out_col_dim,
            conv_23_params.stride, 1, 1, conv_23_params.padding, conv_23_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_22_out, (elem_t*)conv_23_w, (acc_t*)conv_23_b, (elem_t*)conv_23_out,

            RELU, conv_23_params.output_scale,
            conv_23_params.pool_size, 0, conv_23_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 23 cycles: %llu \n", end - start);
    }

    // conv_24
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_24_params.I, conv_24_params.J, conv_24_params.K,
            conv_23_out, conv_24_w, conv_24_b, conv_24_out,
            NO_ACTIVATION, conv_24_params.output_scale, true,
            tiled_matmul_type, check, "conv_24");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 24 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_24_params.I, conv_24_params.J,
        conv_24_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_21_out,
        conv_24_out,
        conv_24_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_25
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_25_params.I, conv_25_params.J, conv_25_params.K,
            conv_24_out, conv_25_w, conv_25_b, conv_25_out,
            RELU, conv_25_params.output_scale, true,
            tiled_matmul_type, check, "conv_25");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 25 cycles: %llu \n", end - start);
    }

    // conv_26
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_25_params.I, conv_25_params.J,
            conv_26_params.I, conv_26_params.K,
            conv_25_out, conv_26_in, &conv_26_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_26_params.I, conv_26_params.J, conv_26_params.K,
            conv_26_in, conv_26_w, conv_26_b, conv_26_out,
            RELU, conv_26_params.output_scale, true,
            tiled_matmul_type, check, "conv_26");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_26_params.batch_size, conv_26_params.in_row_dim, conv_26_params.in_col_dim,
            conv_26_params.in_channels,
            conv_26_params.out_channels, conv_26_params.out_row_dim, conv_26_params.out_col_dim,
            conv_26_params.stride, 1, 1, conv_26_params.padding, conv_26_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_25_out, (elem_t*)conv_26_w, (acc_t*)conv_26_b, (elem_t*)conv_26_out,

            RELU, conv_26_params.output_scale,
            conv_26_params.pool_size, 0, conv_26_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 26 cycles: %llu \n", end - start);
    }

    // conv_27
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_27_params.I, conv_27_params.J, conv_27_params.K,
            conv_26_out, conv_27_w, conv_27_b, conv_27_out,
            NO_ACTIVATION, conv_27_params.output_scale, true,
            tiled_matmul_type, check, "conv_27");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 27 cycles: %llu \n", end - start);
    }

    // Downsampling conv_24_out
    // conv_28
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_24_params.I, conv_24_params.J,
            conv_28_params.I, conv_28_params.K,
            conv_24_out, conv_28_in, &conv_28_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_28_params.I, conv_28_params.J, conv_28_params.K,
            conv_28_in, conv_28_w, conv_28_b, conv_28_out,
            NO_ACTIVATION, conv_28_params.output_scale, true,
            tiled_matmul_type, check, "conv_28");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        // tiled_conv_auto(
        tiled_conv_downsample(
            conv_28_params.batch_size, conv_28_params.in_row_dim, conv_28_params.in_col_dim,
            conv_28_params.in_channels,
            conv_28_params.out_channels, conv_28_params.out_row_dim, conv_28_params.out_col_dim,
            // conv_28_params.stride, 1, 1, conv_28_params.padding, conv_28_params.kernel_size,
            // false, false, false, false, false,

            (elem_t*)conv_24_out, (elem_t*)conv_28_w, (acc_t*)conv_28_b, (elem_t*)conv_28_out,

            NO_ACTIVATION, conv_28_params.output_scale,
            // conv_28_params.pool_size, 0, conv_28_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 28 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_27_params.I, conv_27_params.J,
        conv_27_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_28_out,
        conv_27_out,
        conv_27_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_29
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_27_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_29_params.I, conv_29_params.J, conv_29_params.K,
            conv_27_out, conv_29_w, conv_29_b, conv_29_out,
            RELU, conv_29_params.output_scale, true,
            tiled_matmul_type, check, "conv_29");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 29 cycles: %llu \n", end - start);
    }

    // conv_30
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_29_params.I, conv_29_params.J,
            conv_30_params.I, conv_30_params.K,
            conv_29_out, conv_30_in, &conv_30_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_30_params.I, conv_30_params.J, conv_30_params.K,
            conv_30_in, conv_30_w, conv_30_b, conv_30_out,
            RELU, conv_30_params.output_scale, true,
            tiled_matmul_type, check, "conv_30");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_30_params.batch_size, conv_30_params.in_row_dim, conv_30_params.in_col_dim,
            conv_30_params.in_channels,
            conv_30_params.out_channels, conv_30_params.out_row_dim, conv_30_params.out_col_dim,
            conv_30_params.stride, 1, 1, conv_30_params.padding, conv_30_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_29_out, (elem_t*)conv_30_w, (acc_t*)conv_30_b, (elem_t*)conv_30_out,

            RELU, conv_30_params.output_scale,
            conv_30_params.pool_size, 0, conv_30_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 30 cycles: %llu \n", end - start);
    }

    // conv_31
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            NO_ACTIVATION, conv_31_params.output_scale, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_31_params.I, conv_31_params.J, conv_31_params.K,
            conv_30_out, conv_31_w, conv_31_b, conv_31_out,
            NO_ACTIVATION, conv_31_params.output_scale, true,
            tiled_matmul_type, check, "conv_31");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 31 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_31_params.I, conv_31_params.J,
        conv_31_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_27_out,
        conv_31_out,
        conv_31_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_32
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_31_out, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, true,
            tiled_matmul_type, check, "conv_32");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_32_params.I, conv_32_params.J, conv_32_params.K,
            conv_31_out, conv_32_w, conv_32_b, conv_32_out,
            RELU, conv_32_params.output_scale, true,
            tiled_matmul_type, check, "conv_32");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 32 cycles: %llu \n", end - start);
    }

    // conv_33
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_32_params.I, conv_32_params.J,
            conv_33_params.I, conv_33_params.K,
            conv_32_out, conv_33_in, &conv_33_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_33_params.I, conv_33_params.J, conv_33_params.K,
            conv_33_in, conv_33_w, conv_33_b, conv_33_out,
            RELU, conv_33_params.output_scale, true,
            tiled_matmul_type, check, "conv_33");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_33_params.batch_size, conv_33_params.in_row_dim, conv_33_params.in_col_dim,
            conv_33_params.in_channels,
            conv_33_params.out_channels, conv_33_params.out_row_dim, conv_33_params.out_col_dim,
            conv_33_params.stride, 1, 1, conv_33_params.padding, conv_33_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_32_out, (elem_t*)conv_33_w, (acc_t*)conv_33_b, (elem_t*)conv_33_out,

            RELU, conv_33_params.output_scale,
            conv_33_params.pool_size, 0, conv_33_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 33 cycles: %llu \n", end - start);
    }

    // conv_34
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            NO_ACTIVATION, conv_34_params.output_scale, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_34_params.I, conv_34_params.J, conv_34_params.K,
            conv_33_out, conv_34_w, conv_34_b, conv_34_out,
            NO_ACTIVATION, conv_34_params.output_scale, true,
            tiled_matmul_type, check, "conv_34");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 34 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_34_params.I, conv_34_params.J,
        conv_34_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_31_out,
        conv_34_out,
        conv_34_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_35
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, true,
            tiled_matmul_type, check, "conv_35");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_35_params.I, conv_35_params.J, conv_35_params.K,
            conv_34_out, conv_35_w, conv_35_b, conv_35_out,
            RELU, conv_35_params.output_scale, true,
            tiled_matmul_type, check, "conv_35");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 35 cycles: %llu \n", end - start);
    }

    // conv_36
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_35_params.I, conv_35_params.J,
            conv_36_params.I, conv_36_params.K,
            conv_35_out, conv_36_in, &conv_36_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_36_params.I, conv_36_params.J, conv_36_params.K,
            conv_36_in, conv_36_w, conv_36_b, conv_36_out,
            RELU, conv_36_params.output_scale, true,
            tiled_matmul_type, check, "conv_36");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_36_params.batch_size, conv_36_params.in_row_dim, conv_36_params.in_col_dim,
            conv_36_params.in_channels,
            conv_36_params.out_channels, conv_36_params.out_row_dim, conv_36_params.out_col_dim,
            conv_36_params.stride, 1, 1, conv_36_params.padding, conv_36_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_35_out, (elem_t*)conv_36_w, (acc_t*)conv_36_b, (elem_t*)conv_36_out,

            RELU, conv_36_params.output_scale,
            conv_36_params.pool_size, 0, conv_36_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 36 cycles: %llu \n", end - start);
    }

    // conv_37
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            NO_ACTIVATION, conv_37_params.output_scale, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_37_params.I, conv_37_params.J, conv_37_params.K,
            conv_36_out, conv_37_w, conv_37_b, conv_37_out,
            NO_ACTIVATION, conv_37_params.output_scale, true,
            tiled_matmul_type, check, "conv_37");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 37 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_37_params.I, conv_37_params.J,
        conv_37_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_34_out,
        conv_37_out,
        conv_37_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_38
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_37_out, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, true,
            tiled_matmul_type, check, "conv_38");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_38_params.I, conv_38_params.J, conv_38_params.K,
            conv_37_out, conv_38_w, conv_38_b, conv_38_out,
            RELU, conv_38_params.output_scale, true,
            tiled_matmul_type, check, "conv_38");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 38 cycles: %llu \n", end - start);
    }

    // conv_39
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_38_params.I, conv_38_params.J,
            conv_39_params.I, conv_39_params.K,
            conv_38_out, conv_39_in, &conv_39_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_39_params.I, conv_39_params.J, conv_39_params.K,
            conv_39_in, conv_39_w, conv_39_b, conv_39_out,
            RELU, conv_39_params.output_scale, true,
            tiled_matmul_type, check, "conv_39");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_39_params.batch_size, conv_39_params.in_row_dim, conv_39_params.in_col_dim,
            conv_39_params.in_channels,
            conv_39_params.out_channels, conv_39_params.out_row_dim, conv_39_params.out_col_dim,
            conv_39_params.stride, 1, 1, conv_39_params.padding, conv_39_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_38_out, (elem_t*)conv_39_w, (acc_t*)conv_39_b, (elem_t*)conv_39_out,

            RELU, conv_39_params.output_scale,
            conv_39_params.pool_size, 0, conv_39_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 39 cycles: %llu \n", end - start);
    }

    // conv_40
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            NO_ACTIVATION, conv_40_params.output_scale, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_40_params.I, conv_40_params.J, conv_40_params.K,
            conv_39_out, conv_40_w, conv_40_b, conv_40_out,
            NO_ACTIVATION, conv_40_params.output_scale, true,
            tiled_matmul_type, check, "conv_40");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 40 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_40_params.I, conv_40_params.J,
        conv_40_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_37_out,
        conv_40_out,
        conv_40_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_41
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_41_params.I, conv_41_params.J, conv_41_params.K,
            conv_40_out, conv_41_w, conv_41_b, conv_41_out,
            RELU, conv_41_params.output_scale, true,
            tiled_matmul_type, check, "conv_41");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 41 cycles: %llu \n", end - start);
    }

    // conv_42
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_41_params.I, conv_41_params.J,
            conv_42_params.I, conv_42_params.K,
            conv_41_out, conv_42_in, &conv_42_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_42_params.I, conv_42_params.J, conv_42_params.K,
            conv_42_in, conv_42_w, conv_42_b, conv_42_out,
            RELU, conv_42_params.output_scale, true,
            tiled_matmul_type, check, "conv_42");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_42_params.batch_size, conv_42_params.in_row_dim, conv_42_params.in_col_dim,
            conv_42_params.in_channels,
            conv_42_params.out_channels, conv_42_params.out_row_dim, conv_42_params.out_col_dim,
            conv_42_params.stride, 1, 1, conv_42_params.padding, conv_42_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_41_out, (elem_t*)conv_42_w, (acc_t*)conv_42_b, (elem_t*)conv_42_out,

            RELU, conv_42_params.output_scale,
            conv_42_params.pool_size, 0, conv_42_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 42 cycles: %llu \n", end - start);
    }

    // conv_43
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            NO_ACTIVATION, conv_43_params.output_scale, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_43_params.I, conv_43_params.J, conv_43_params.K,
            conv_42_out, conv_43_w, conv_43_b, conv_43_out,
            NO_ACTIVATION, conv_43_params.output_scale, true,
            tiled_matmul_type, check, "conv_43");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 43 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_43_params.I, conv_43_params.J,
        conv_43_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_40_out,
        conv_43_out,
        conv_43_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_44
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_43_out, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, true,
            tiled_matmul_type, check, "conv_44");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_44_params.I, conv_44_params.J, conv_44_params.K,
            conv_43_out, conv_44_w, conv_44_b, conv_44_out,
            RELU, conv_44_params.output_scale, true,
            tiled_matmul_type, check, "conv_44");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 44 cycles: %llu \n", end - start);
    }

    // conv_45
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_44_params.I, conv_44_params.J,
            conv_45_params.I, conv_45_params.K,
            conv_44_out, conv_45_in, &conv_45_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_45_params.I, conv_45_params.J, conv_45_params.K,
            conv_45_in, conv_45_w, conv_45_b, conv_45_out,
            RELU, conv_45_params.output_scale, true,
            tiled_matmul_type, check, "conv_45");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_45_params.batch_size, conv_45_params.in_row_dim, conv_45_params.in_col_dim,
            conv_45_params.in_channels,
            conv_45_params.out_channels, conv_45_params.out_row_dim, conv_45_params.out_col_dim,
            conv_45_params.stride, 1, 1, conv_45_params.padding, conv_45_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_44_out, (elem_t*)conv_45_w, (acc_t*)conv_45_b, (elem_t*)conv_45_out,

            RELU, conv_45_params.output_scale,
            conv_45_params.pool_size, 0, conv_45_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 45 cycles: %llu \n", end - start);
    }

    // conv_46
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            NO_ACTIVATION, conv_46_params.output_scale, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_46_params.I, conv_46_params.J, conv_46_params.K,
            conv_45_out, conv_46_w, conv_46_b, conv_46_out,
            NO_ACTIVATION, conv_46_params.output_scale, true,
            tiled_matmul_type, check, "conv_46");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 46 cycles: %llu \n", end - start);
    }

    // Downsampling conv_43_out
    // conv_47
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_43_params.I, conv_43_params.J,
            conv_47_params.I, conv_47_params.K,
            conv_43_out, conv_47_in, &conv_47_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_47_params.I, conv_47_params.J, conv_47_params.K,
            conv_47_in, conv_47_w, conv_47_b, conv_47_out,
            NO_ACTIVATION, conv_47_params.output_scale, true,
            tiled_matmul_type, check, "conv_47");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_47_params.batch_size, conv_47_params.in_row_dim, conv_47_params.in_col_dim,
            conv_47_params.in_channels,
            conv_47_params.out_channels, conv_47_params.out_row_dim, conv_47_params.out_col_dim,
            conv_47_params.stride, 1, 1, conv_47_params.padding, conv_47_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_43_out, (elem_t*)conv_47_w, (acc_t*)conv_47_b, (elem_t*)conv_47_out,

            NO_ACTIVATION, conv_47_params.output_scale,
            conv_47_params.pool_size, 0, conv_47_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 47 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_46_params.I, conv_46_params.J,
        conv_46_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_47_out,
        conv_46_out,
        conv_46_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_48
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_46_out, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_48_params.I, conv_48_params.J, conv_48_params.K,
            conv_46_out, conv_48_w, conv_48_b, conv_48_out,
            RELU, conv_48_params.output_scale, true,
            tiled_matmul_type, check, "conv_48");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 48 cycles: %llu \n", end - start);
    }

    // conv_49
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_48_params.I, conv_48_params.J,
            conv_49_params.I, conv_49_params.K,
            conv_48_out, conv_49_in, &conv_49_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_49_params.I, conv_49_params.J, conv_49_params.K,
            conv_49_in, conv_49_w, conv_49_b, conv_49_out,
            RELU, conv_49_params.output_scale, true,
            tiled_matmul_type, check, "conv_49");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_49_params.batch_size, conv_49_params.in_row_dim, conv_49_params.in_col_dim,
            conv_49_params.in_channels,
            conv_49_params.out_channels, conv_49_params.out_row_dim, conv_49_params.out_col_dim,
            conv_49_params.stride, 1, 1, conv_49_params.padding, conv_49_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_48_out, (elem_t*)conv_49_w, (acc_t*)conv_49_b, (elem_t*)conv_49_out,

            RELU, conv_49_params.output_scale,
            conv_49_params.pool_size, 0, conv_49_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 49 cycles: %llu \n", end - start);
    }

    // conv_50
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_49_out, conv_50_w, conv_50_b, conv_50_out,
            NO_ACTIVATION, conv_50_params.output_scale, true,
            tiled_matmul_type, check, "conv_50");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_50_params.I, conv_50_params.J, conv_50_params.K,
            conv_49_out, conv_50_w, conv_50_b, conv_50_out,
            NO_ACTIVATION, conv_50_params.output_scale, true,
            tiled_matmul_type, check, "conv_50");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 50 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_50_params.I, conv_50_params.J,
        conv_50_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_46_out,
        conv_50_out,
        conv_50_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;
    
    // conv_51
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_51_params.I, conv_51_params.J, conv_51_params.K,
            conv_50_out, conv_51_w, conv_51_b, conv_51_out,
            RELU, conv_51_params.output_scale, true,
            tiled_matmul_type, check, "conv_51");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 51 cycles: %llu \n", end - start);
    }

    // conv_52
    if (!conv) {
      start = read_cycles();

        im2col_with_col2im(conv_51_params.I, conv_51_params.J,
            conv_52_params.I, conv_52_params.K,
            conv_51_out, conv_52_in, &conv_52_params);

        end = read_cycles();
        im2col_cycles += end - start;

        start = read_cycles();

        tiled_matmul_nn_auto(conv_52_params.I, conv_52_params.J, conv_52_params.K,
            conv_52_in, conv_52_w, conv_52_b, conv_52_out,
            RELU, conv_52_params.output_scale, true,
            tiled_matmul_type, check, "conv_52");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_conv_auto(
            conv_52_params.batch_size, conv_52_params.in_row_dim, conv_52_params.in_col_dim,
            conv_52_params.in_channels,
            conv_52_params.out_channels, conv_52_params.out_row_dim, conv_52_params.out_col_dim,
            conv_52_params.stride, 1, 1, conv_52_params.padding, conv_52_params.kernel_size,
            false, false, false, false, false,

            (elem_t*)conv_51_out, (elem_t*)conv_52_w, (acc_t*)conv_52_b, (elem_t*)conv_52_out,

            RELU, conv_52_params.output_scale,
            conv_52_params.pool_size, 0, conv_52_params.pool_padding,

            tiled_matmul_type);

        end = read_cycles();
        conv_cycles += end - start;
        printf("conv 52 cycles: %llu \n", end - start);
    }

    // conv_53
    if (!conv) {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            NO_ACTIVATION, conv_53_params.output_scale, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;

    } else {
        start = read_cycles();

        tiled_matmul_nn_auto(conv_53_params.I, conv_53_params.J, conv_53_params.K,
            conv_52_out, conv_53_w, conv_53_b, conv_53_out,
            NO_ACTIVATION, conv_53_params.output_scale, true,
            tiled_matmul_type, check, "conv_53");

        end = read_cycles();
        matmul_cycles += end - start;
        printf("matmul 53 cycles: %llu \n", end - start);
    }

    // Add residuals
    start = read_cycles();

    tiled_resadd_auto(conv_53_params.I, conv_53_params.J,
        conv_53_params.res_scale,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        conv_50_out,
        conv_53_out,
        conv_53_out,
        true,
        tiled_matmul_type == CPU ? CPU : WS);

    end = read_cycles();
    res_add_cycles += end - start;

    // Global averaging
    static elem_t average[4][2048] row_align(1);

    start = read_cycles();

    tiled_global_average_auto(conv_53_out, average, conv_53_params.batch_size,
        conv_53_params.out_channels, conv_53_params.out_row_dim, CPU);

    end = read_cycles();
    other_cycles += end - start;

    // fc_54
    start = read_cycles();

    tiled_matmul_nn_auto(fc_54_params.I, fc_54_params.J, fc_54_params.K,
        average, fc_54_w, fc_54_b, fc_54_out,
        NO_ACTIVATION, fc_54_params.output_scale, false,
        tiled_matmul_type, check, "fc_54");

    end = read_cycles();
    matmul_cycles += end - start;
    printf("matmul 54 cycles: %llu \n", end - start);

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

    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_cycles + conv_dw_cycles + res_add_cycles + other_cycles+1;

    printf("\nTotal cycles: %llu (100%%)\n", total_cycles);
    printf("Matmul cycles: %llu (%d%%)\n", matmul_cycles, (matmul_cycles * 100) / total_cycles);
    printf("Im2col cycles: %llu (%d%%)\n", im2col_cycles, (im2col_cycles * 100) / total_cycles);
    printf("Conv cycles: %llu (%d%%)\n", conv_cycles, (conv_cycles * 100) / total_cycles);
    printf("Pooling cycles: %llu (%d%%)\n", pool_cycles, (pool_cycles * 100) / total_cycles);
    printf("Depthwise convolution cycles: %llu (%d%%)\n", conv_dw_cycles, (conv_dw_cycles * 100) / total_cycles);
    printf("Res add cycles: %llu (%d%%)\n", res_add_cycles, (res_add_cycles * 100) / total_cycles);
    printf("Other cycles: %llu (%d%%)\n", other_cycles, (other_cycles * 100) / total_cycles);

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

