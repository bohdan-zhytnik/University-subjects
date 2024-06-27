#ifndef __QUEUE_H__
#define __QUEUE_H__

// #include "queue.c"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// typedef struct queue_t queue_t;


typedef struct {
    void **array;
    int head;
    int tail;
    int size;
    int capacity;
} queue_t;

queue_t* create_queue(int capacity);
void delete_queue(queue_t *queue);
bool push_to_queue(queue_t *queue, void *data);
void* pop_from_queue(queue_t *queue);
void* get_from_queue(queue_t *queue, int idx);
int get_queue_size(queue_t *queue);

#endif /* __QUEUE_H__ */


