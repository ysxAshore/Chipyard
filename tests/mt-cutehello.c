#include <riscv-pk/encoding.h>
#include <stdio.h>
#include "marchid.h"
#include "mt-thread.h"
#include "mt-queuelock.h"
#include "resetnet50_static_graph.h"

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
volatile int exit_ok = 0;


task_pool *pool = NULL;

void calculate_server(void){

    while(execute_task(pool))
    {
      ;
    }
  
    while(1)
    {
      //waiting for interrupt
      asm volatile (
        "wfi"
      );
    }

}

void init_calculate_thread(void){

    //配软中断的陷入地址，参考/root/chipyard-main/chipyard/toolchains/libgloss/misc/crt0.S就行
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

  



  //let core 0 know that we are finished
  __sync_fetch_and_add(&exit_ok, 1);



}



int main(void) {
  //this is main thread
  size_t mhartid = read_csr(mhartid);
  const char* march = get_march(read_csr(marchid));
  printf("Hello YJP, there is main thread.\n");
  printf("Main thred is core[marchid=%d] mhartid=%lu, a %s.\n",read_csr(marchid), mhartid, march);

  //TODO：linux会读取某个文件来获取模型图信息

  //fence
  __sync_synchronize();
  //初始化任务池
  pool = task_pool_init(128);
  resnet50_graph *graph = init_resnet50_graph();
  graph_node *current_node = graph->head;


  //解析模型文件，往任务池里添加任务
  void (*task_function)(int thread_id, void *thread_params);
  //for task in nn_graph;
  while((task_function = get_current_task(graph, current_node)) != NULL)
  {
    
    task_function = get_current_task(graph, current_node);

    if (task_function != NULL) {
        task_pool_add_task(pool, task_function, (void *)(&current_node->conv_param));
    }
    //异步启动：软中断使计算线程执行任务
    //同步：确认当前可并行图节点任务已完成
    current_node = get_next_node(graph, current_node);
    if (current_node == NULL) {
      break;
    }
  }
  //end for

  init_down = 1;
  __sync_synchronize();
  //模型解析完毕,仿真结束

  //wait exit ok
    while (exit_ok != n_cores-1) {
        ;
    }

  return 0;
}
