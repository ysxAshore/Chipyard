#include<stdio.h>
#include<stdlib.h>
//输入卷积层的输入(n,ih,iw,ic)的数据排布
//输入卷积核大小kernel_size,卷积核步长conv_stride(kh,kw,oc,ic)的数据排布
//输入卷积层的输出(n,oh,ow,oc)的数据排布

int main(int argc, char *argv[])
{
    //argc != 10，输出提示信息，并重新输入
    if (argc != 10)
    {
        printf("argc should be 10\n");
        //提示要输入的参数
        printf("input n ih iw ic kernel_size conv_stride oh ow oc\n");
        return 0;
    }
    int n,ih,iw,ic;
    int kernel_size,conv_stride;
    int oh,ow,oc;

    //读取参数
    n = atoi(argv[1]);
    ih = atoi(argv[2]);
    iw = atoi(argv[3]);
    ic = atoi(argv[4]);
    kernel_size = atoi(argv[5]);
    conv_stride = atoi(argv[6]);
    oh = atoi(argv[7]);
    ow = atoi(argv[8]);
    oc = atoi(argv[9]);
    

    //如果输入的参数不合法，输出提示信息，并重新输入
    if (n<1||ih<1||iw<1||ic<1||kernel_size<1||conv_stride<1||oh<1||ow<1||oc<1)
    {
        printf("n,ih,iw,ic,kernel_size,conv_stride,oh,ow,oc should be larger than 0\n");
        return 0;
    }
    //n必须等于1
    if (n!=1)
    {
        printf("n must be 1\n");
        return 0;
    }
    //ih必须等于iw，且oh*conv_stride必须等于ih，且ow必须等于oh，ic和oc必须是64的倍数
    if (ih!=iw||oh*conv_stride!=ih||ow!=oh||ic%64!=0||oc%64!=0)
    {
        printf("ih must equal to iw,oh*conv_stride must equal to ih,ow must equal to oh,ic and oc must be multiple of 64\n");
        return 0;
    }
    //如果mnk小于32，输出提示信息，并重新输入
    int input_m = n*ih*iw;
    int input_k = ic;
    int weight_n = kernel_size*kernel_size*oc;
    int weight_k = ic;
    int output_m = n*oh*ow;
    int output_n = oc;

    int **input,**weight,**output,*bias;
    input = (int **)malloc(input_m*sizeof(int *));
    weight = (int **)malloc(weight_n*sizeof(int *));
    output = (int **)malloc(output_m*sizeof(int *));
    bias = (int *)malloc(output_m*sizeof(int));
    for (int i = 0; i < input_m; i++)
    {
        input[i] = (int *)malloc(input_k*sizeof(int));
    }
    for (int i = 0; i < weight_n; i++)
    {
        weight[i] = (int *)malloc(weight_k*sizeof(int));
    }
    for (int i = 0; i < output_m; i++)
    {
        output[i] = (int *)malloc(output_n*sizeof(int));
    }

    for (int i = 0; i < input_m; i++)
    {
        for (int j = 0; j < input_k; j++)
        {
            input[i][j] = rand()%256-128;
        }
        
    }

    for (int i = 0; i < weight_n; i++)
    {
        for (int j = 0; j < weight_k; j++)
        {
            weight[i][j] = rand()%256-128;
        }
        
    }

    //bias
    for (int i = 0; i < output_n; i++)
    {
        bias[i] = rand()%256-128;
    }

    for (int patch_i = 0;patch_i<n;patch_i++)
    {
        for(int oh_i = 0;oh_i<oh;oh_i++)
        {
            for(int ow_i = 0;ow_i<ow;ow_i++)
            {
                for(int oc_i = 0;oc_i<oc;oc_i++)
                {
                    int temp_acc = 0;
                    for (int kh_i = -kernel_size/2; kh_i <= kernel_size/2; kh_i++)
                    {
                        for (int kw_i = -kernel_size/2; kw_i <= kernel_size/2; kw_i++)
                        {
                            int ih_i = oh_i*conv_stride+kh_i;
                            int iw_i = ow_i*conv_stride+kw_i;
                            if (ih_i<0||ih_i>=ih||iw_i<0||iw_i>=iw)
                            {
                                continue;
                            }
                            int input_index = patch_i*ih*iw+ih_i*iw+iw_i;
                            int weight_index = ((kh_i+kernel_size/2)*kernel_size+kw_i+kernel_size/2)*oc+oc_i;
                            for (int ic_i = 0; ic_i < ic; ic_i++)
                            {
                                temp_acc += input[input_index][ic_i]*weight[weight_index][ic_i];
                            }
                        }
                    }
                    output[patch_i*oh*ow+oh_i*ow+ow_i][oc_i] = temp_acc+bias[oc_i];
                }
            }
        }
    }
    //输入文件名
    char filename[100];
    //覆盖写文件
    FILE *fp = fopen("conv_value.h","w");

    //输出mnk，application_m等，用宏输出
    fprintf(fp,"#define APPLICATION_M %d\n",output_m);
    fprintf(fp,"#define APPLICATION_N %d\n",output_n);
    fprintf(fp,"#define APPLICATION_K %d\n",weight_k);
    fprintf(fp,"#define BIAS_TYPE %d\n",2);
    //输出注释，1表示zero bias，2表示repeat row bias，3表示full bias
    fprintf(fp,"//1:zero bias,2:repeat row bias,3:full bias\n");
    //输出conv_stride,kernel_size,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max
    //输出conv_stride = 1,kernel_size = 1,kernel_stride = 0,因为矩阵乘不需要这些值
    fprintf(fp,"#define CONV_STRIDE %d\n",conv_stride);
    fprintf(fp,"#define KERNEL_SIZE %d\n",kernel_size);
    fprintf(fp,"#define KERNEL_STRIDE %d\n",ic*oc);
    //stride_A = K,stride_B = K,stride_C = 4*N,stride_D = 4*N
    fprintf(fp,"#define STRIDE_A %d\n",weight_k);
    fprintf(fp,"#define STRIDE_B %d\n",weight_k);
    fprintf(fp,"#define STRIDE_C %d\n",4*output_n);
    fprintf(fp,"#define STRIDE_D %d\n",4*output_n);
    
    //transpose_result = 0,conv_oh_index = 0,conv_ow_index = 0,conv_oh_max = 1,conv_ow_max = M
    fprintf(fp,"#define TRANSPOSE_RESULT 0\n");
    fprintf(fp,"#define CONV_OH_INDEX 0\n");
    fprintf(fp,"#define CONV_OW_INDEX 0\n");
    fprintf(fp,"#define CONV_OH_PER_ADD %d\n",64/ow);
    fprintf(fp,"#define CONV_OW_PER_ADD %d\n",64%ow);
    fprintf(fp,"#define CONV_OH_MAX %d\n",oh);
    fprintf(fp,"#define CONV_OW_MAX %d\n",ow);
    // void * VectorOp,int VectorInst_Length代表了要融合的向量任务的具体指令和指令块长度。


// 矩阵乘就是IH=1，IW=M，IC=K，OC=N，KH=1，KW=1，STRIDE=1的卷积

    int i,j;
    fprintf(fp,"static char input[%d][%d] __attribute__((aligned(256))) = {\n",input_m,input_k);
    for ( i = 0; i < input_m; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < input_k; j++)
        {
            fprintf(fp,"%d,",input[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static char weight[%d][%d] __attribute__((aligned(256))) = {\n",weight_n,weight_k);
    for ( i = 0; i < weight_n; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < weight_k; j++)
        {
            fprintf(fp,"%d,",weight[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static int gloden_output[%d][%d] __attribute__((aligned(256))) = {\n",output_m,output_n);
    for ( i = 0; i < output_m; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < output_n; j++)
        {
            fprintf(fp,"%d,",output[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    
    fprintf(fp,"static int bias[%d] __attribute__((aligned(256))) = {\n",output_n);
    // fprintf(fp,"{");
    for ( j = 0; j < output_n; j++)
    {
        fprintf(fp,"%d,",bias[j]);
    }
    fprintf(fp,"};\n");


    fprintf(fp,"static int output[%d][%d] __attribute__((aligned(256))) = {\n",output_m,output_n);
    for ( i = 0; i < output_m; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < output_n; j++)
        {
            fprintf(fp,"%d,",0);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");

    fclose(fp);
    return 0;
}
