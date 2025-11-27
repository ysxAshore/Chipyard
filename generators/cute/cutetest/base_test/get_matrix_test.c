#include<stdio.h>
#include<stdlib.h>
//输入矩阵大小(M,N,K)，数据类型(int8),偏置的方式，输出矩阵乘的随机输入,并输出正确结果

int main()
{
    int m,n,k;
    printf("input m,n,k:");
    scanf("%d %d %d",&m,&n,&k);
    //如果mnk小于32，输出提示信息，并重新输入
    if (m<32||n<32||k<32)
    {
        printf("m,n,k should be larger than 32\n");
        return 0;
    }
    int type;
    printf("input type(1:int8,zero bias,2:int8 repeat row bias,3:int8 full bias):");
    scanf("%d",&type);
    int **a,**b,**c,**d;
    a = (int **)malloc(m*sizeof(int *));
    b = (int **)malloc(k*sizeof(int *));
    c = (int **)malloc(m*sizeof(int *));
    d = (int **)malloc(m*sizeof(int *));
    for (int i = 0; i < m; i++)
    {
        a[i] = (int *)malloc(k*sizeof(int));
        c[i] = (int *)malloc(n*sizeof(int));
        d[i] = (int *)malloc(n*sizeof(int));
    }
    for (int i = 0; i < k; i++)
    {
        b[i] = (int *)malloc(n*sizeof(int));
    }
    int i,j,l;
    for ( i = 0; i < m; i++)
    {
        for ( j = 0; j < k; j++)
        {
            a[i][j] = rand()%256-128;
        }
        
    }
    for ( i = 0; i < n; i++)
    {
        for ( j = 0; j < k; j++)
        {
            b[i][j] = rand()%256-128;
        }
        
    }
    if (type==1)
    {
        for ( i = 0; i < m; i++)
        {
            for ( j = 0; j < n; j++)
            {
                c[i][j] = 0;
                for ( l = 0; l < k; l++)
                {
                    c[i][j] += a[i][l]*b[j][l];
                }
                // printf("%d %d %d\n",i,j,l);
            }
            
        }
        
    }
    else if (type==2)
    {
        for ( i = 0; i < m; i++)
        {
            for ( j = 0; j < n; j++)
            {
                if(i==0)
                {
                    d[0][j] = rand()%893465 - 446732;
                }
                else
                {
                    d[i][j] = d[0][j];
                }
                for ( l = 0; l < k; l++)
                {
                    c[i][j] += a[i][l]*b[j][l];
                }
                // printf("%d %d %d\n",i,j,l);
                c[i][j] += d[i][j];
            }
        }
    }
    else if (type==3)
    {
        for ( i = 0; i < m; i++)
        {
            for ( j = 0; j < n; j++)
            {
                d[i][j] = rand()%893465 - 446732;
                for ( l = 0; l < k; l++)
                {
                    c[i][j] += a[i][l]*b[j][l];
                }
                // printf("%d %d %d\n",i,j,l);
                c[i][j] += d[i][j];
            }
        }
    }

    //输入是否要转置矩阵
    int transpose;
    printf("transpose result?(1:yes,0:no):");
    scanf("%d",&transpose);
    //输出a,b,c,d矩阵。
    //将a,b,c,d矩阵的值输出到.h文件中，用于测试

    //输入文件名
    char filename[100];
    //覆盖写文件
    FILE *fp = fopen("matmul_value.h","w");

    //输出mnk，application_m等，用宏输出
    fprintf(fp,"#define APPLICATION_M %d\n",m);
    fprintf(fp,"#define APPLICATION_N %d\n",n);
    fprintf(fp,"#define APPLICATION_K %d\n",k);
    fprintf(fp,"#define BIAS_TYPE %d\n",type);
    //输出注释，1表示zero bias，2表示repeat row bias，3表示full bias
    fprintf(fp,"//1:zero bias,2:repeat row bias,3:full bias\n");
    //输出conv_stride,kernel_size,kernel_stride,stride_A,stride_B,stride_C,stride_D,transpose_result,conv_oh_index,conv_ow_index,conv_oh_max,conv_ow_max
    //输出conv_stride = 1,kernel_size = 1,kernel_stride = 0,因为矩阵乘不需要这些值
    fprintf(fp,"#define CONV_STRIDE 1\n");
    fprintf(fp,"#define KERNEL_SIZE 1\n");
    fprintf(fp,"#define KERNEL_STRIDE 0\n");
    //stride_A = K,stride_B = K,stride_C = 4*N,stride_D = 4*N
    fprintf(fp,"#define STRIDE_A %d\n",k);
    fprintf(fp,"#define STRIDE_B %d\n",k);
    fprintf(fp,"#define STRIDE_C %d\n",4*n);
    fprintf(fp,"#define STRIDE_D %d\n",4*n);
    
    //transpose_result = 0,conv_oh_index = 0,conv_ow_index = 0,conv_oh_max = 1,conv_ow_max = M
    fprintf(fp,"#define TRANSPOSE_RESULT %d\n",transpose);
    fprintf(fp,"#define CONV_OH_INDEX 0\n");
    fprintf(fp,"#define CONV_OW_INDEX 0\n");
    fprintf(fp,"#define CONV_OH_MAX 1\n");
    fprintf(fp,"#define CONV_OW_MAX %d\n",m);
    // void * VectorOp,int VectorInst_Length代表了要融合的向量任务的具体指令和指令块长度。


// 矩阵乘就是IH=1，IW=M，IC=K，OC=N，KH=1，KW=1，STRIDE=1的卷积

    fprintf(fp,"static char a[%d][%d] __attribute__((aligned(256))) = {\n",m,k);
    for ( i = 0; i < m; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < k; j++)
        {
            fprintf(fp,"%d,",a[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    fprintf(fp,"static char b[%d][%d] __attribute__((aligned(256))) = {\n",n,k);
    for ( i = 0; i < n; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < k; j++)
        {
            fprintf(fp,"%d,",b[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");
    if(transpose == 0)
    {
        fprintf(fp,"static int gloden_c[%d][%d] __attribute__((aligned(256))) = {\n",m,n);
        for ( i = 0; i < m; i++)
        {
            fprintf(fp,"{");
            for ( j = 0; j < n; j++)
            {
                fprintf(fp,"%d,",c[i][j]);
            }
            fprintf(fp,"},\n");
        }
        fprintf(fp,"};\n");
        
        fprintf(fp,"static int c[%d][%d] __attribute__((aligned(256))) = {\n",m,n);
        for ( i = 0; i < m; i++)
        {
            fprintf(fp,"{");
            for ( j = 0; j < n; j++)
            {
                fprintf(fp,"%d,",0);
            }
            fprintf(fp,"},\n");
        }
        fprintf(fp,"};\n");
    }else
    {
        fprintf(fp,"static int gloden_c[%d][%d] __attribute__((aligned(256))) = {\n",n,m);
        for ( i = 0; i < n; i++)
        {
            fprintf(fp,"{");
            for ( j = 0; j < m; j++)
            {
                fprintf(fp,"%d,",c[j][i]);
            }
            fprintf(fp,"},\n");
        }
        fprintf(fp,"};\n");
    
        fprintf(fp,"static int c[%d][%d] __attribute__((aligned(256))) = {\n",n,m);
        for ( i = 0; i < n; i++)
        {
            fprintf(fp,"{");
            for ( j = 0; j < m; j++)
            {
                fprintf(fp,"%d,",0);
            }
            fprintf(fp,"},\n");
        }
        fprintf(fp,"};\n");
    }

    fprintf(fp,"static int d[%d][%d] __attribute__((aligned(256))) = {\n",m,n);
    for ( i = 0; i < m; i++)
    {
        fprintf(fp,"{");
        for ( j = 0; j < n; j++)
        {
            fprintf(fp,"%d,",d[i][j]);
        }
        fprintf(fp,"},\n");
    }
    fprintf(fp,"};\n");


    // //生成warm_up三个数组的函数，目标让数组都在cache中
    // //让这个函数不要被优化
    // //每次都跨256bit读一个数
    // int stride_int8 = 256/8/sizeof(char);
    // int stride_int16 = 256/8/sizeof(short);
    // int stride_int32 = 256/8/sizeof(int);
    // int stride;
    // if (type==1||type==0)
    // {
    //     stride = stride_int8;
    // }
    // else if (type==2)
    // {
    //     stride = stride_int16;
    // }
    // else if (type==3)
    // {
    //     stride = stride_int32;
    // }
    // fprintf(fp,"__attribute__((optimize(\"O0\"))) void warm_up()\n{\n");
    // fprintf(fp,"    int i,j;\n");
    // fprintf(fp,"    int t;\n");
    // fprintf(fp,"    for ( i = 0; i < %d; i++)\n",m);
    // fprintf(fp,"    {\n");
    // fprintf(fp,"        for ( j = 0; j < %d; j+= %d)\n",k,stride);
    // fprintf(fp,"        {\n");
    // fprintf(fp,"            a[i][j] = a[i][j];\n");
    // fprintf(fp,"        }\n");
    // fprintf(fp,"    }\n");
    // fprintf(fp,"    for ( i = 0; i < %d; i++)\n",k);
    // fprintf(fp,"    {\n");
    // fprintf(fp,"        for ( j = 0; j < %d; j+= %d)\n",n,stride);
    // fprintf(fp,"        {\n");
    // fprintf(fp,"            b[i][j] = b[i][j];\n");
    // fprintf(fp,"        }\n");
    // fprintf(fp,"    }\n");
    // fprintf(fp,"    for ( i = 0; i < %d; i++)\n",m);
    // fprintf(fp,"    {\n");
    // fprintf(fp,"        for ( j = 0; j < %d; j += %d)\n",n,stride_int32);
    // fprintf(fp,"        {\n");
    // fprintf(fp,"            c[i][j] = c[i][j];\n");
    // fprintf(fp,"        }\n");
    // fprintf(fp,"    }\n");
    // fprintf(fp,"    for ( i = 0; i < %d; i++)\n",m);
    // fprintf(fp,"    {\n");
    // fprintf(fp,"        for ( j = 0; j < %d; j += %d)\n",n,stride_int32);
    // fprintf(fp,"        {\n");
    // fprintf(fp,"            d[i][j] = d[i][j];\n");
    // fprintf(fp,"        }\n");
    // fprintf(fp,"    }\n");
    // fprintf(fp,"}\n");

    // //还需要一个固定的把所有L1的数据刷新掉的程序
    // fprintf(fp,"__attribute__((optimize(\"O0\"))) void flush_cache()\n{\n");
    // //申请64KB的连续数组256位对齐，不要用malloc
    // fprintf(fp,"    char a[64*1024] __attribute__((aligned(256)));\n");
    // fprintf(fp,"    int i;\n");
    // //每次跨256bit读一个数，来填满L1D
    // fprintf(fp,"    for ( i = 0; i < 64*1024; i+=32)\n");
    // fprintf(fp,"    {\n");
    // fprintf(fp,"        a[i] = a[i];\n");
    // fprintf(fp,"    }\n");
    // fprintf(fp,"}\n");

    fclose(fp);
    return 0;
}
