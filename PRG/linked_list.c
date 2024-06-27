#include <stdio.h>
#include <stdlib.h>

struct Node{
    int payload;
    struct Node * next;
    struct Node * prior;
};

struct LinkedList {
    struct Node * head;  
    struct Node * tail; 
};


struct LinkedList * LinkedList_new(){
    struct LinkedList * ll = (struct LinkedList *)malloc(sizeof(struct LinkedList)); // ll - lokalni promenna(ukazatel) - pamet na steku a v ni je pamet z heapu
    ll -> head = NULL;
    ll -> tail = NULL;
    return ll; // ne znika protoze je to ukazatel
}

int LinkedList_push(struct LinkedList * llp, int payload){
    
    if (llp == NULL){ // kontrola existence linked listu
        return 0;
    }
    
    struct Node * np = malloc(sizeof(struct Node));
    if (np == NULL){
        return 0;
    }

    np->payload = payload;
    np->next = NULL;

    if (llp->head == NULL){
        llp-> head = np;
        np->prior=NULL;
    }
    // if (llp->head != NULL){

    // }
    if (llp->tail == NULL){
        llp->tail = np;
        // if (llp-> head != llp->tail){
        //     llp->tail->prior=llp-> head;
        // }
    }
    else{
        struct Node * previous_tail = llp->tail;
        previous_tail->next = np;
        np->prior= previous_tail;
    }
    llp->tail = np;
    return 1;
}

void LinkedList_print(struct LinkedList * llp){
    struct Node * current_node = llp->head;

    while (current_node != NULL){
        printf("%d ", current_node->payload);
        current_node = current_node->next;
    } 
}

void LinkedList_print_reverse_order(struct LinkedList * llp){
    struct Node * current_node = llp->tail;

    while (current_node != NULL){
        printf("%d ", current_node->payload);
        current_node = current_node->prior;
    } 
}


void LinkedList_destroy(struct LinkedList * llp){
    struct Node * current_node = llp->head;

    while(current_node != NULL){
        struct Node * next_node = current_node->next;
        free(current_node);
        current_node = next_node;
    }

    free(llp);
}


int main(){
    struct LinkedList * ll1 = LinkedList_new();

    LinkedList_push(ll1, 42);
    LinkedList_push(ll1, 32);
    LinkedList_push(ll1, 48);

    LinkedList_print(ll1);
    putchar('\n');
    LinkedList_print_reverse_order(ll1);
    putchar('\n');

    LinkedList_destroy(ll1);
    ll1 = NULL;

    return 0;
}


