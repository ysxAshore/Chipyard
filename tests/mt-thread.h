#include "mt-queuelock.h"
#include <stdlib.h>
#ifndef MT_THREAD_H
#define MT_THREAD_H

//任务池相关结构体和函数

//任务相关结构体和函数
typedef struct thread_task{
    void (*task_function)(int thread_id, void *thread_params);
    // int thread_id;
    void *thread_params;
    // bool is_done;
}thread_task;

typedef struct task_pool{
    thread_task *tasks;
    int size;
    int capacity;
    loop_ticket_t ticket;
}task_pool;

task_pool *task_pool_init(int capacity) {
    task_pool *pool = (task_pool *)malloc(sizeof(task_pool));
    pool->tasks = (thread_task *)malloc(sizeof(thread_task) * capacity);
    pool->size = 0;
    pool->capacity = capacity;
    pool->ticket.max = capacity;
    pool->ticket.next_ticket = 0;
    return pool;
}

void task_pool_add_task(task_pool *pool, void (*task_function)(int, void*), void *thread_params) {
    // queue_lock_acquire(&pool->insert_task_lock);//目前默认一个核可以添加任务
    if (pool->size < pool->capacity) {
        pool->tasks[pool->size].task_function = task_function;
        pool->tasks[pool->size].thread_params = thread_params;
        pool->size++;
        pool->ticket.max = pool->size;
    }
    // queue_lock_release(&pool->insert_task_lock);
}

int execute_task(task_pool *pool) {
    int my_ticket = loop_ticket_acquire(&pool->ticket);
    if (my_ticket == -1) {
        return 0;//finished
    }
    //执行任务
    pool->tasks[my_ticket].task_function(my_ticket, pool->tasks[my_ticket].thread_params);
    return 1;//haven't finished
    
}


#endif
