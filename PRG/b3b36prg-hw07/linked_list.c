#include <stdbool.h>
#include <stdlib.h>


typedef struct Node
{
    int value;
    struct Node * next;
}Node;

Node * tail = NULL;
Node * head = NULL;

_Bool push(int entry){
    if (entry < 0) { return false; }
    Node * new_node=malloc(sizeof(Node));
    if (new_node == NULL){ return false; }
    new_node->value=entry;
    new_node->next = NULL;
    if (head == NULL){ head = tail = new_node; }
    else{
        tail->next=new_node;
        tail = new_node;
    }
    return true;
}
int pop(void){
    if ( head == NULL){
        return -1;
    }
    Node * tmp = head;
    int value = head->value;
    head=head->next;
    free(tmp);
    return value;
}

_Bool insert(int entry){
    if (entry < 0) { return false; }
    Node * node_insert=malloc(sizeof(Node));
    if (node_insert == NULL){ return false; }
    node_insert->value=entry;
    node_insert->next=NULL;

    if (head == NULL){
        head = tail = node_insert;
        return true;
    }

    Node *N_next= NULL;
    if (node_insert->value >= head->value){
        node_insert->next = head;
        head = node_insert;
        return true;
    }else{
        Node *prior = head;
        N_next = head->next;
    
        while(N_next != NULL){
            // if (node_insert->value >= head->value){
            //     node_insert->next = head;
            //     head = node_insert;
            // }
            // if (N_next->next == NULL){

            // }
            if (node_insert->value >= N_next->value){
                node_insert->next = N_next;
                N_next = node_insert;
                prior->next=N_next;
                return true;
            }else{
                prior = N_next;
                N_next = N_next->next;
            }
        }
        N_next=node_insert;
        tail = N_next;
        prior->next=N_next; 
        return true;
    }
}

_Bool erase(int entry){
    Node *curr = head;
    Node *prior = NULL;
    _Bool ret = false;
    while (curr != NULL){
        if (curr->value == entry){
            if (prior != NULL){
                prior->next=curr->next;
            }else{
                head = curr->next;
            }
            if (curr->next == NULL){
                tail = prior;
            }
            Node *temp = curr;
            curr = curr->next;
            free(temp);
            ret = true;
        } else {
            prior = curr;
            curr = curr->next;
        }
        
    }
    return ret;
}











































// typedef struct Node {
//     int value;
//     struct Node *next;
// } Node;

// Node *head = NULL;
// Node *tail = NULL;

// _Bool push(int entry) {
//     if (entry < 0) return false;
//     Node *new_node = (Node *)malloc(sizeof(Node));
//     if (!new_node) return false;

//     new_node->value = entry;
//     new_node->next = NULL;

//     if (!head) {
//         head = tail = new_node;
//     } else {
//         tail->next = new_node;
//         tail = new_node;
//     }
//     return true;
// }

// int pop(void) {
//     if (!head) return -1;

//     Node *temp = head;
//     int value = temp->value;

//     head = head->next;
//     free(temp);

//     return value;
// }

// _Bool insert(int entry) {
//     if (entry < 0) return false;
//     Node *new_node = (Node *)malloc(sizeof(Node));
//     if (!new_node) return false;

//     new_node->value = entry;
//     new_node->next = NULL;

//     if (!head) {
//         head = tail = new_node;
//         return true;
//     }

//     if (entry >= head->value) {
//         new_node->next = head;
//         head = new_node;
//         return true;
//     }

//     Node *curr = head;
//     while (curr->next && entry < curr->next->value) {
//         curr = curr->next;
//     }

//     new_node->next = curr->next;
//     curr->next = new_node;
//     if (!new_node->next) tail = new_node;
//     return true;
// }

// _Bool erase(int entry) {
//     Node *curr = head;
//     Node *prev = NULL;
//     _Bool removed = false;

//     while (curr) {
//         if (curr->value == entry) {
//             if (prev) {
//                 prev->next = curr->next;
//             } else {
//                 head = curr->next;
//             }

//             if (!curr->next) tail = prev;

//             Node *temp = curr;
//             curr = curr->next;
//             free(temp);
//             removed = true;
//         } else {
//             prev = curr;
//             curr = curr->next;
//         }
//     }
//     return removed;
// }

int getEntry(int idx) {
    Node *curr = head;
    int i = 0;
    while ((curr!=NULL) && (i < idx)) {
        curr = curr->next;
        i++;
    }
    if (curr!=NULL){
        return curr->value;
    }else
        return -1;
    // return curr ? curr->value : -1;
}

int size(void) {
    int count = 0;
    Node *curr = head;
    while (curr != NULL) {
        count++;
        curr = curr->next;
    }
    return count;
}

void clear(void) {
    Node *curr = head;
    while (curr != NULL) {
        Node *temp = curr;
        curr = curr->next;
        free(temp);
    }
    head = tail = NULL;
}


