#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
// #include "include/gemmini.h"
// #include "include/gemmini_nn.h"

#include "resnet50_params_int8_quick.h"
#include "images.h"
#include <stdlib.h>

#define PRINT_CONV_WEIGHTS(conv, dim1, dim2) \
    printf("static const elem_t " #conv "_w_t[%d][%d] row_align(1) = {", dim1, dim2); \
    for (int i = 0; i < conv##_params.kernel_size; i++) { \
        for (int j = 0; j < conv##_params.kernel_size; j++) { \
            for (int l = 0; l < conv##_params.out_channels; l++) { \
                printf("{"); \
                for (int k = 0; k < conv##_params.in_channels; k++) { \
                    printf("%d", conv##_w[i * conv##_params.kernel_size * conv##_params.in_channels + j * conv##_params.in_channels + k][l]); \
                    if (k != conv##_params.in_channels - 1) \
                        printf(","); \
                } \
                printf("},"); \
            } \
        } \
    } \
    printf("};\n");

int main (int argc, char * argv[]) {

    
    // PRINT_CONV_WEIGHTS(conv_2, conv_2_params.out_channels * conv_2_params.kernel_size * conv_2_params.kernel_size, conv_2_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_3, conv_3_params.out_channels * conv_3_params.kernel_size * conv_3_params.kernel_size, conv_3_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_4, conv_4_params.out_channels * conv_4_params.kernel_size * conv_4_params.kernel_size, conv_4_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_5, conv_5_params.out_channels * conv_5_params.kernel_size * conv_5_params.kernel_size, conv_5_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_6, conv_6_params.out_channels * conv_6_params.kernel_size * conv_6_params.kernel_size, conv_6_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_7, conv_7_params.out_channels * conv_7_params.kernel_size * conv_7_params.kernel_size, conv_7_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_8, conv_8_params.out_channels * conv_8_params.kernel_size * conv_8_params.kernel_size, conv_8_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_9, conv_9_params.out_channels * conv_9_params.kernel_size * conv_9_params.kernel_size, conv_9_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_10, conv_10_params.out_channels * conv_10_params.kernel_size * conv_10_params.kernel_size, conv_10_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_11, conv_11_params.out_channels * conv_11_params.kernel_size * conv_11_params.kernel_size, conv_11_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_12, conv_12_params.out_channels * conv_12_params.kernel_size * conv_12_params.kernel_size, conv_12_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_13, conv_13_params.out_channels * conv_13_params.kernel_size * conv_13_params.kernel_size, conv_13_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_14, conv_14_params.out_channels * conv_14_params.kernel_size * conv_14_params.kernel_size, conv_14_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_15, conv_15_params.out_channels * conv_15_params.kernel_size * conv_15_params.kernel_size, conv_15_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_16, conv_16_params.out_channels * conv_16_params.kernel_size * conv_16_params.kernel_size, conv_16_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_17, conv_17_params.out_channels * conv_17_params.kernel_size * conv_17_params.kernel_size, conv_17_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_18, conv_18_params.out_channels * conv_18_params.kernel_size * conv_18_params.kernel_size, conv_18_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_19, conv_19_params.out_channels * conv_19_params.kernel_size * conv_19_params.kernel_size, conv_19_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_20, conv_20_params.out_channels * conv_20_params.kernel_size * conv_20_params.kernel_size, conv_20_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_21, conv_21_params.out_channels * conv_21_params.kernel_size * conv_21_params.kernel_size, conv_21_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_22, conv_22_params.out_channels * conv_22_params.kernel_size * conv_22_params.kernel_size, conv_22_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_23, conv_23_params.out_channels * conv_23_params.kernel_size * conv_23_params.kernel_size, conv_23_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_24, conv_24_params.out_channels * conv_24_params.kernel_size * conv_24_params.kernel_size, conv_24_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_25, conv_25_params.out_channels * conv_25_params.kernel_size * conv_25_params.kernel_size, conv_25_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_26, conv_26_params.out_channels * conv_26_params.kernel_size * conv_26_params.kernel_size, conv_26_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_27, conv_27_params.out_channels * conv_27_params.kernel_size * conv_27_params.kernel_size, conv_27_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_28, conv_28_params.out_channels * conv_28_params.kernel_size * conv_28_params.kernel_size, conv_28_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_29, conv_29_params.out_channels * conv_29_params.kernel_size * conv_29_params.kernel_size, conv_29_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_30, conv_30_params.out_channels * conv_30_params.kernel_size * conv_30_params.kernel_size, conv_30_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_31, conv_31_params.out_channels * conv_31_params.kernel_size * conv_31_params.kernel_size, conv_31_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_32, conv_32_params.out_channels * conv_32_params.kernel_size * conv_32_params.kernel_size, conv_32_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_33, conv_33_params.out_channels * conv_33_params.kernel_size * conv_33_params.kernel_size, conv_33_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_34, conv_34_params.out_channels * conv_34_params.kernel_size * conv_34_params.kernel_size, conv_34_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_35, conv_35_params.out_channels * conv_35_params.kernel_size * conv_35_params.kernel_size, conv_35_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_36, conv_36_params.out_channels * conv_36_params.kernel_size * conv_36_params.kernel_size, conv_36_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_37, conv_37_params.out_channels * conv_37_params.kernel_size * conv_37_params.kernel_size, conv_37_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_38, conv_38_params.out_channels * conv_38_params.kernel_size * conv_38_params.kernel_size, conv_38_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_39, conv_39_params.out_channels * conv_39_params.kernel_size * conv_39_params.kernel_size, conv_39_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_40, conv_40_params.out_channels * conv_40_params.kernel_size * conv_40_params.kernel_size, conv_40_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_41, conv_41_params.out_channels * conv_41_params.kernel_size * conv_41_params.kernel_size, conv_41_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_42, conv_42_params.out_channels * conv_42_params.kernel_size * conv_42_params.kernel_size, conv_42_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_43, conv_43_params.out_channels * conv_43_params.kernel_size * conv_43_params.kernel_size, conv_43_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_44, conv_44_params.out_channels * conv_44_params.kernel_size * conv_44_params.kernel_size, conv_44_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_45, conv_45_params.out_channels * conv_45_params.kernel_size * conv_45_params.kernel_size, conv_45_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_46, conv_46_params.out_channels * conv_46_params.kernel_size * conv_46_params.kernel_size, conv_46_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_47, conv_47_params.out_channels * conv_47_params.kernel_size * conv_47_params.kernel_size, conv_47_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_48, conv_48_params.out_channels * conv_48_params.kernel_size * conv_48_params.kernel_size, conv_48_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_49, conv_49_params.out_channels * conv_49_params.kernel_size * conv_49_params.kernel_size, conv_49_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_50, conv_50_params.out_channels * conv_50_params.kernel_size * conv_50_params.kernel_size, conv_50_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_51, conv_51_params.out_channels * conv_51_params.kernel_size * conv_51_params.kernel_size, conv_51_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_52, conv_52_params.out_channels * conv_52_params.kernel_size * conv_52_params.kernel_size, conv_52_params.in_channels);
    // PRINT_CONV_WEIGHTS(conv_53, conv_53_params.out_channels * conv_53_params.kernel_size * conv_53_params.kernel_size, conv_53_params.in_channels);
    // PRINT_CONV_WEIGHTS(fc_54, 1000, 2048);
    printf("static const elem_t fc_54_w_t[1000][2048] row_align(1) = {");
    for (int j = 0;j<1000;j++){
        printf("{");
        for (int i = 0;i<2048;i++)
        {
            printf("%d",fc_54_w[i][j]);
            if (i!=2047)
                printf(",");
        }
        printf("},");
    }
    printf("};\n");


    exit(0);
}

