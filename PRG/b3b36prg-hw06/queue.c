#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "queue.h"

queue_t* create_queue(int capacity);
void delete_queue(queue_t *queue);
bool push_to_queue(queue_t *queue, void *data);
void* pop_from_queue(queue_t *queue);
void* get_from_queue(queue_t *queue, int idx);
int get_queue_size(queue_t *queue);

queue_t* create_queue(int capacity){
    queue_t *queue=malloc(sizeof(queue_t));
    queue->array=malloc(capacity*sizeof(void*));
    queue->head=0;
    queue->tail=0;
    queue->size = 0;
    queue->capacity=capacity;
    return queue;
}

void delete_queue(queue_t *queue){
    free(queue->array);
    free(queue);
}

bool push_to_queue(queue_t *queue, void *data){
    if (data==NULL){
        return false;
    }
    if ((queue->capacity)==queue->size){
        int old_capacity=queue->capacity;
        queue->capacity*=2;
        void **new_array=malloc(queue->capacity*(sizeof(void*)));
        if (new_array==NULL){
            return false;
        } 
        for (int i=0; i < queue->size; i++){
            new_array[i]=queue->array[((queue->head+i) % old_capacity)];
        }
        free(queue->array);
        queue->array=new_array;
        queue->head=0;
        queue->tail=queue->size;
    }
    if ((queue->capacity)==queue->size) {
        return false;
    }
    queue->array[queue->tail]=data;
    queue->tail=(queue->tail+1)%queue->capacity;
    queue->size++;
    return true;
}

void* pop_from_queue(queue_t *queue){
    if (queue->size==0){
        return NULL;
    }
    void *value=queue->array[queue->head];
    queue->head=(queue->head+1)%queue->capacity;
    queue->size--;
    if (queue->size > 0 && queue->size < ((queue->capacity/3)*2)-1){
        int old_capacity=queue->capacity;
        queue->capacity=((queue->capacity/3)*2)+1;
        void **new_array=malloc(queue->capacity*(sizeof(void*)));
        if (new_array==NULL){
            return NULL;
        } 
        for (int i=0; i < queue->size; i++){
            new_array[i]=queue->array[((queue->head+i) % old_capacity)];
        }
        free(queue->array);
        queue->array=new_array;
        queue->head=0;
        queue->tail=queue->size; 
    }
    return value;
}

void* get_from_queue(queue_t *queue, int idx){
    void *value = NULL;
    if (idx < 0 || idx >= queue->size){
        return NULL;
    }
    value = queue->array[(queue->head+idx)%queue->capacity];

    return value;
}

int get_queue_size(queue_t *queue){
    return queue->size;
}

