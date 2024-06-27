
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK EXIT_SUCCESS
#define ERROR_INPUT 100
#define ERROR_LENGHT 101

unsigned char * read_input(int *);
void shift (unsigned char *,int len);
int compare(unsigned char *, unsigned char *,int len);

int main() {
    unsigned char* str_encoded = NULL;
    unsigned char* str_real = NULL;
    int len_str_encoded = 0;
    int len_str_real = 0;
    str_encoded = read_input(&len_str_encoded);
    str_real = read_input(&len_str_real);

    if (!((str_encoded!=NULL) && (str_real!=NULL))){
        fprintf(stderr,"Error: Chybny vstup!\n");
            free(str_encoded);
            free(str_real);
        return ERROR_INPUT;
    }

    if (len_str_encoded != len_str_real){
                free(str_encoded);
                free(str_real);
        fprintf(stderr,"Error: Chybna delka vstupu!\n");
        return ERROR_LENGHT;
    }

    int return_offset=compare(str_encoded, str_real, len_str_encoded);

    if (return_offset != 0){
        for (int i=0; i< return_offset; i++){
            shift(str_encoded, len_str_encoded);
        }
    }

    printf("%s\n",str_encoded);

    free(str_encoded);
    free(str_real);
    
    return OK;
}

void shift (unsigned char *str1, int len){
    for (int i=0; i< len; i++){

        if (str1[i]=='z'){
            str1[i]='A';
        }
        else if (str1[i]=='Z'){
            str1[i]='a';
        }else{
        str1[i]++;
        }
    }
}

int compare(unsigned char *str1, unsigned char *str2,int len){
    int offset=0;
    int best_count=0;
    int best_offset=0;
    for (int j = 0; j < 53; j++){
        int count=0;
        //the number of matches between the arrays
        
        if (offset != 0){      
            shift(str1, len);
        }

        for (int i=0; i< len; i++){
            if (str1[i]==str2[i]){
                count++;
            }
        }
        if (best_count < count){
            best_count = count;
            best_offset = offset;
        }
        offset++;
        if (best_count == len){
            int return_offset = 0;
            return return_offset;
        }
    }
    int return_offset = best_offset;
    return return_offset;


}

unsigned char * read_input(int * size){
    int allocated_memory_capacity = 3;
    unsigned char * memorry=malloc(allocated_memory_capacity*sizeof(unsigned char));
    if (memorry == NULL){
        return NULL;
    }
  int count_of_loaded_letters = 0;
  char c_l_l = 0;
    while(1){
        c_l_l = getchar();

        if (c_l_l=='\n'){
            memorry[count_of_loaded_letters]='\0';
            //I don't know why I don't have '/0' at the end of the string.
            *size =  count_of_loaded_letters;
            return memorry;
        }

        if (!((c_l_l >= 'a' && c_l_l <= 'z') || (c_l_l >= 'A' && c_l_l <= 'Z') || (c_l_l=='\n'))) {
            free(memorry);
            return NULL;
        }

        if(count_of_loaded_letters>=allocated_memory_capacity-1){
            allocated_memory_capacity=allocated_memory_capacity*2;
            unsigned char * new_memorry = realloc(memorry, allocated_memory_capacity*sizeof(unsigned char));
            if (new_memorry == NULL){
                free(memorry);
                return NULL;
            }
            memorry = new_memorry;
        }
        memorry[count_of_loaded_letters]=c_l_l;
        count_of_loaded_letters++;
    }

}



