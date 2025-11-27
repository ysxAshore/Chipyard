#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "resnet50_params_cpu.h"
// #include "resnet50_params_1batch.h"
#include "images.h"
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define CONV_IN(layer) conv_##layer##_in
#define CONV_OUT(layer) conv_##layer##_out
#define CONV_W(layer) conv_##layer##_w
#define CONV_B(layer) conv_##layer##_b
#define CONV_PARAMS(layer) conv_##layer##_params

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define ARRAY_WIDTH(arr) (sizeof((arr)[0]) / sizeof((arr)[0][0]))


void tiled_conv_cute(struct ConvParams, const elem_t * input, const elem_t * weights, const acc_t * bias, elem_t * output, int act, acc_scale_t scale)
{
    //所有数据都按照channel-major的顺序存储
    //input     = [batch_size, in_h, in_w, in_channels]
    //weights   = [kernel_h, kernel_w, out_channels, in_channels] ---> 方便从某个h,w开始读取一个oc,ic的矩阵,用于分块矩阵乘
    //bias      = [out_channels] ---> repeat row Load
    //output    = [batch_size, out_h, out_w, out_channels] ---> 保持和input一样的输入内存布局

    //如果是kernel_size=1，stride=1的卷积层，直接分块矩阵乘

    //如果是kernel_size=1，stride=2的卷积层，直接分块矩阵乘，调整矩阵乘的row stride就行了

    //如果是kernel_size=3，stride=1的卷积层，分块，(OS)9Load->1Store矩阵乘，TensorLoad自动补0

    //如果是kernel_size=3，stride=2的卷积层，分块，(OS)9Load->1Store矩阵乘，TensorLoad自动补0

    //TODO:next_step1: L1D优化的卷积

    //TODO:next_step2: Easy Scale的卷积

    //TODO:next_step3: Vector Control的卷积

    



}

#define PRINT_LAYER(layer) \
    printf("CONV_PARAMS(%d) = {",layer); \
    printf("\tih iw ic = %d %d %d, ",CONV_PARAMS(layer).in_row_dim,CONV_PARAMS(layer).in_col_dim,CONV_PARAMS(layer).in_channels); \
    printf("\toh ow oc = %d %d %d, ",CONV_PARAMS(layer).out_row_dim,CONV_PARAMS(layer).out_col_dim,CONV_PARAMS(layer).out_channels); \
    printf("\tkh kw ic oc = %d %d %d %d, ",CONV_PARAMS(layer).kernel_size,CONV_PARAMS(layer).kernel_size,CONV_PARAMS(layer).in_channels,CONV_PARAMS(layer).out_channels); \
    printf("\tstride = %d, ",CONV_PARAMS(layer).stride); \
    printf("};");\
    printf("\t%d(%d,%d,%d)%d,padding=%d,%d%%%d==%d ",CONV_PARAMS(layer).kernel_size,CONV_PARAMS(layer).out_row_dim*CONV_PARAMS(layer).out_col_dim,CONV_PARAMS(layer).out_channels,CONV_PARAMS(layer).in_channels,CONV_PARAMS(layer).stride,CONV_PARAMS(layer).padding,CONV_PARAMS(layer).out_row_dim*CONV_PARAMS(layer).out_col_dim,64,(CONV_PARAMS(layer).out_row_dim*CONV_PARAMS(layer).out_col_dim)%64); \
    printf("\tInput_size:%.2fMB,weight_size:%.2fMB,output_size:%.2fMB,Mops:%.2fMMAC\n",CONV_PARAMS(layer).in_row_dim*CONV_PARAMS(layer).in_col_dim*CONV_PARAMS(layer).in_channels/1024.0/1024.0,CONV_PARAMS(layer).kernel_size*CONV_PARAMS(layer).kernel_size*CONV_PARAMS(layer).in_channels*CONV_PARAMS(layer).out_channels/1024.0/1024.0,CONV_PARAMS(layer).out_row_dim*CONV_PARAMS(layer).out_col_dim*CONV_PARAMS(layer).out_channels*4/1024.0/1024.0,CONV_PARAMS(layer).out_row_dim*CONV_PARAMS(layer).out_col_dim*CONV_PARAMS(layer).out_channels*CONV_PARAMS(layer).kernel_size*CONV_PARAMS(layer).kernel_size*CONV_PARAMS(layer).in_channels/1000.0/1000.0);





int main() {
    PRINT_LAYER(2);
    PRINT_LAYER(3);
    PRINT_LAYER(4);
    PRINT_LAYER(5);
    PRINT_LAYER(6);
    PRINT_LAYER(7);
    PRINT_LAYER(8);
    PRINT_LAYER(9);
    PRINT_LAYER(10);
    PRINT_LAYER(11);
    PRINT_LAYER(12);
    PRINT_LAYER(13);
    PRINT_LAYER(14);
    PRINT_LAYER(15);
    PRINT_LAYER(16);
    PRINT_LAYER(17);
    PRINT_LAYER(18);
    PRINT_LAYER(19);
    PRINT_LAYER(20);
    PRINT_LAYER(21);
    PRINT_LAYER(22);
    PRINT_LAYER(23);
    PRINT_LAYER(24);
    PRINT_LAYER(25);
    PRINT_LAYER(26);
    PRINT_LAYER(27);
    PRINT_LAYER(28);
    PRINT_LAYER(29);
    PRINT_LAYER(30);
    PRINT_LAYER(31);
    PRINT_LAYER(32);
    PRINT_LAYER(33);
    PRINT_LAYER(34);
    PRINT_LAYER(35);
    PRINT_LAYER(36);
    PRINT_LAYER(37);
    PRINT_LAYER(38);
    PRINT_LAYER(39);
    PRINT_LAYER(40);
    PRINT_LAYER(41);
    PRINT_LAYER(42);
    PRINT_LAYER(43);
    PRINT_LAYER(44);
    PRINT_LAYER(45);
    PRINT_LAYER(46);
    PRINT_LAYER(47);
    PRINT_LAYER(48);
    PRINT_LAYER(49);
    PRINT_LAYER(50);
    PRINT_LAYER(51);
    PRINT_LAYER(52);
    PRINT_LAYER(53);





    return 0;
}

enum tiled_matmul_type_t {OS, WS, CPU};


