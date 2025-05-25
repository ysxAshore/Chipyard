#include <riscv-pk/encoding.h>
#include <stdio.h>
#include <stdint.h>
#include "marchid.h"
#include "mt-thread.h"
#include "mt-queuelock.h"
#include "resetnet50_static_graph.h"

#define CLINT_BASE  0x02000000

// EDIT THIS
static size_t n_cores = 3;

static void __attribute__((noinline)) barrier()
{
  static volatile int sense;
  static volatile int count;
  static __thread int threadsense;

  __sync_synchronize();

  threadsense = !threadsense;
  if (__sync_fetch_and_add(&count, 1) == n_cores-1)
  {
    count = 0;
    sense = threadsense;
  }
  else while(sense != threadsense)
    ;

  __sync_synchronize();
}

// static queue_lock_t hello_lock;
// void init_locks() {
//     queue_lock_init(&hello_lock);
// }

volatile int init_down = 0;
volatile int ready = 0;
volatile int execute_ok = 0;

task_pool *pool = NULL;

void calculate_server(void){

  // Clear MSIP for current hart

  uint32_t hartid;
  __asm__ volatile ("csrr %0, mhartid" : "=r"(hartid));
  volatile uint32_t* msip_reg = (volatile uint32_t*)(CLINT_BASE + (hartid << 2));
  *msip_reg = 0;  // 直接写内存，等效于清除 MSIP

  while(execute_task(pool))
  {
    ;
  }
    
  /* 
    Notify globally,
    current core finish tasks, 
  */
  
  __sync_fetch_and_add(&execute_ok, 1);   

  // Enable Global Interrupts

  __asm__ volatile ("csrs mstatus, %0" : : "r" (0x8));

  // Enable Software Interrupts

  __asm__ volatile ("csrs mie, %0" : : "r" (0x8));

  while(1)
  {
    //waiting for interrupt
    asm volatile (
      "wfi"
    );
  }

}

void init_calculate_thread(void){

  //设置其他核心的软中断地址为 calculate_server

  uintptr_t trap_addr = (uintptr_t)calculate_server;
  write_csr(mtvec, trap_addr);

  // Enable Global Interrupts

  __asm__ volatile ("csrs mstatus, %0" : : "r" (0x8));

  // Enable Software Interrupts
  
  __asm__ volatile ("csrs mie, %0" : : "r" (0x8));

}

void __main(void) {
  //this is __main thread,caculate core is here

  //设置中断地址为__main,或者core0设置的位置

  //wait init_down
  while (init_down == 0) {
        ;
  }

  init_calculate_thread();

  size_t mhartid = read_csr(mhartid);
  const char* march = get_march(read_csr(marchid));
  
  queue_lock_acquire(&console_lock);
  printf("Hello YJP, there is calculate thread.\n");
  printf("Calculate thread is core[marchid=%d] mhartid=%lu, a %s\n",read_csr(marchid), mhartid, march);
  queue_lock_release(&console_lock);

  /* 
    Notify globally,
    current core finish tasks, 
  */
  
  __sync_fetch_and_add(&ready, 1);   

  while(1)
  {
    //waiting for interrupt
    asm volatile (
      "wfi"
    );
  }

}

int main(void) {
  //this is main thread
  size_t mhartid = read_csr(mhartid);
  const char* march = get_march(read_csr(marchid));
  printf("Hello YJP, there is main thread.\n");
  printf("Main thred is core[marchid=%d] mhartid=%lu, a %s.\n",read_csr(marchid), mhartid, march);

  //TODO：linux会读取某个文件来获取模型图信息

  //fence

  //初始化任务池
  pool = task_pool_init(128);
  init_down = 1;
  __sync_synchronize();

  // 同步：所有进程可以执行图节点任务
  while(!__sync_bool_compare_and_swap(&ready, n_cores-1, 0)){
    ;
  }

  resnet50_graph *graph = init_resnet50_graph();
  graph_node *current_node = graph->head;

  //解析模型文件，往任务池里添加任务
  void (*task_function)(int thread_id, void *thread_params);

  //for task in nn_graph;

  /* 
    TODO: get_cuurent_task return a list
    Add tasks in list into pool
  */
  while((task_function = get_current_task(graph, current_node)) != NULL)
  {
    // Add tasks into pool as a batch
    int batch_size = 2;
    for (int i=0; i<batch_size; i++){
      task_pool_add_task(pool, task_function, (void *)(&current_node->conv_param));
    }

    //异步启动：软中断使计算线程执行任务
    for(size_t target_hartid=1; target_hartid<n_cores; target_hartid++){
      volatile uint32_t* msip_reg = (volatile uint32_t*)(CLINT_BASE + (target_hartid << 2));
      *msip_reg = 1;  // 直接写内存，置位MSIP
    }
    __sync_synchronize();
    
    //同步：确认当前可并行图节点任务已完成
    while(!__sync_bool_compare_and_swap(&execute_ok, n_cores-1, 0)){
      ;
    }
    
    current_node = get_next_node(graph, current_node);
    if (current_node == NULL) {
      break;
    }

  }
  //end for

  //模型解析完毕,仿真结束

  return 0;
}
