#include <riscv-pk/encoding.h>
#include <stdio.h>
#include "marchid.h"

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

//排队锁
typedef struct {
    volatile int next_ticket;
    volatile int now_serving;
} queue_lock_t;
void queue_lock_init(queue_lock_t *lock) {
    lock->next_ticket = 0;
    lock->now_serving = 0;
}
void queue_lock_acquire(queue_lock_t *lock) {
    int my_ticket = __sync_fetch_and_add(&lock->next_ticket, 1);
    while (lock->now_serving != my_ticket) {
        // Busy wait
    }
}
void queue_lock_release(queue_lock_t *lock) {
    __sync_fetch_and_add(&lock->now_serving, 1);
}
static queue_lock_t hello_lock;
void init_locks() {
    queue_lock_init(&hello_lock);
}

volatile int init_down = 0;
volatile int exit_ok = 0;

void init_calculate_thread(void){

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
  queue_lock_acquire(&hello_lock);
  for (size_t i = 0; i < n_cores; i++) {
    if (mhartid == i) {
      printf("Hello YJP, there is calculate thread.\n");
      printf("Calculate thread is core[marchid=%d] mhartid=%lu, a %s\n",read_csr(marchid), mhartid, march);
      queue_lock_release(&hello_lock);
    }
  }
    

  //wait core 0
  __sync_fetch_and_add(&exit_ok, 1);

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

  //fence
  init_locks();
  __sync_synchronize();
  init_down = 1;
  __sync_synchronize();


  //wait exit ok
    while (exit_ok != n_cores-1) {
        ;
    }

  return 0;
}
