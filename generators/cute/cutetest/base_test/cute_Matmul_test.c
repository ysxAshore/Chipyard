#include <stdio.h>
#include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "matmul_value.h"

//1024*1024的256位对齐的char数组
// static char a[256*] __attribute__((aligned(256)));
// static char b[1024*1024] __attribute__((aligned(256)));
// static int  c[1024*1024] __attribute__((aligned(256)));

// val TaskTypeTensorLoad = 2.U(TypeBitWidth.W)
// val TaskTypeTensorZeroLoad = 3.U(TypeBitWidth.W) //直接将数据填充为0，实际上是什么也没做，默认可以写入SRAM，无视以前SRAM里面的数据即可
// val TaskTypeTensorRepeatRowLoad = 4.U(TypeBitWidth.W) //重复加载一行数据，实际上是什么也没做，默认可以写入SRAM，无视以前SRAM里面的数据即可

//让C[0][0~3] = 0,C[0][4~7] = 1, C[0][8~11] = 2, C[0][12~15] = 3, C[0][16~19] = 4, C[0][20~23] = 5, C[0][24~27] = 6, C[0][28~31] = 7,C[0][32~35] = 8, C[0][36~39] = 9, C[0][40~43] = 10, C[0][44~47] = 11, C[0][48~51] = 12, C[0][52~55] = 13, C[0][56~59] = 14, C[0][60~63] = 15
void m_c_init()
{
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            c[0][i*4+j] = i;
        }
    }
}

int main(void) {
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

    m_c_init();
    uint64_t res1 = 1;
    uint64_t A = a;
    uint64_t A_Stride = APPLICATION_K * sizeof(a[0][0]);
    uint64_t B = b;
    uint64_t B_Stride = APPLICATION_K * sizeof(b[0][0]);
    uint64_t C = c;
    uint64_t C_Stride = APPLICATION_N * sizeof(c[0][0]);
    uint64_t D = d;
    uint64_t D_Stride = APPLICATION_N * sizeof(d[0][0]);
    uint64_t element_type = 1;//1byte per input
    uint64_t bias_type = TaskTypeTensorRepeatRowLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;
    uint64_t start_cycle = mrdcycle();
    uint64_t issue_val = issue_cute_matmul_marco_inst(A, A_Stride, B, B_Stride, C, C_Stride, D, D_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K, element_type, bias_type, TRANSPOSE_RESULT, current_M_index);

    printf("issue_val: %ld\n", issue_val);
    //查询指令FIFO的情况
    res1 = cute_marco_inst_fifo_valid_search();
    if(res1){
        printf("FIFO not empty\n");
    }else{
        printf("FIFO empty\n");
        return -1;
    }

    printf("finish\n");
    YGJK_INS_RRR(res1, 0, 0, 2);
	printf("acc time: %ldcycles\n", res1);
    YGJK_INS_RRR(res1, 0, 0, 5);
    printf("compute: %ldcycles\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 3);
	printf("acc read req: %ld\n", res1);
	YGJK_INS_RRR(res1, 0, 0, 4);
	printf("acc write req: %ld\n", res1);

    printf("D Test start\n");
    int error = 0;
    for (int i = 0;i<sizeof(d)/sizeof(int);i++)
    {
        if(*((int*)d+i) != *((int*)c+i))
        {
            printf("[error!] D[%d](%d) not equal C[%d](%d)\n",i,*((int*)d+i),i,*((int*)c+i));
            error = 1;
        }
    }
    if (error == 0)
    {
        printf("all test ok!\n");
    }
    
    

  return 0;
}
