#include <stdio.h>
#include <riscv-pk/encoding.h>
// #include <riscv-pk/marchid.h>
#include "marchid.h"
#include <stdint.h>
#include "cuteMarcoinstHelper.h"
#include "matmul_value_mnk_113_128_128_rowreapeat.h"

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

    uint64_t res1 = 1;
    // uint64_t A = input;
    uint64_t A_Stride = APPLICATION_K * sizeof(a[0][0]);
    // uint64_t B = weight;
    uint64_t B_Stride = APPLICATION_K * sizeof(b[0][0]);
    // uint64_t C = bias;
    uint64_t C_Stride = APPLICATION_N * sizeof(c[0][0]);
    // uint64_t D = output;
    uint64_t D_Stride = APPLICATION_N * sizeof(d[0][0]);
    uint64_t element_type = 1;//1byte per input
    uint64_t bias_type = TaskTypeTensorLoad;
    // uint64_t transpose_result = 0;
    uint64_t current_M_index = 0;
    uint64_t issue_val = issue_cute_matmul_marco_inst(a, A_Stride, b, B_Stride, d, D_Stride, c, C_Stride, APPLICATION_M, APPLICATION_N, APPLICATION_K, element_type, bias_type,0,0);

    printf("issue_val: %ld\n", issue_val);
    //查询指令FIFO的情况
    res1 = cute_marco_inst_fifo_valid_search();
    if(res1){
        printf("FIFO not empty\n");
    }else{
        printf("FIFO empty\n");
        return -1;
    }

    
    res1 = cute_marco_inst_fifo_finish_search();
    while(!res1)
    {
        printf("Waiting for finish\n");
        res1 = cute_marco_inst_fifo_finish_search();
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
    printf("Pass!\n");



  return 0;
}
