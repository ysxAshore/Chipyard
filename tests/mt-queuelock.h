#ifndef MT_QEUEUELOCK_H
#define MT_QEUEUELOCK_H

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

typedef struct {
    volatile int next_ticket;
    int max;
} loop_ticket_t;

void loop_ticket_init(loop_ticket_t *lock) {
    lock->next_ticket = 0;
}

int loop_ticket_acquire(loop_ticket_t *lock) {
    int my_ticket = __sync_fetch_and_add(&lock->next_ticket, 1);
    if(my_ticket < lock->max) return my_ticket;
    else{
        __sync_fetch_and_sub(&lock->next_ticket, 1);
        return -1;
    }
}



#endif
