  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h> 

  #define OK 0
  #define ERROR_INPUT 100

struct Matrix create_matrix(int rows, int cols);
int read_matrix(struct Matrix *matrix);
struct Matrix* get_matrices(int *num_matrices, char *operations);
void print_matrix(struct Matrix *matrix);
int shift(struct Matrix *matrices, int num_matrices, int matrix_index);
int adding_up(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2);
int subtraction(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2);
int multiplication(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2);
size_t my_strlen(char *str);
int operations_execution(struct Matrix *matrices, int num_matrices, char *operations);


struct Matrix {
    int rows;
    int cols;
    int len_matrix;
    int *data;
};

// Function to clear the input buffer
void clear_input_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}


int main() {
  struct Matrix *matrices=NULL;
  int num_matrices=0;     
  // Number of matrices to work with
  char operations[50];
  // Operations to perform on matrices
  matrices=get_matrices(&num_matrices, operations);
  if (matrices == NULL){
    fprintf(stderr,"Error: Chybny vstup!\n");
    return ERROR_INPUT;
  }
  num_matrices=operations_execution(matrices, num_matrices, operations);
  if (num_matrices==0){
    fprintf(stderr,"Error: Chybny vstup!\n");
    free(matrices);
    return ERROR_INPUT;
  } 
  for (int i=0; i<num_matrices; i++){
    printf("%d %d\n",matrices[i].rows,matrices[i].cols);
    print_matrix(&matrices[i]);
  }

  for (int i=0; i<num_matrices; i++){
    free(matrices[i].data);
  }
  free(matrices);
  return 0; 
}

size_t my_strlen(char *str)
{
  size_t length = 0;
  while (str[length] != '\0')
  {
    length++;
  }
  return length;
}

// Function to shift operations in the string to the left
void shift_operetions(char *operations,int index){
  for (int i=index+1; i< my_strlen(operations); i++){
    operations[i-1]=operations[i];
  }
  operations[my_strlen(operations)-1]='\0';
}
int operations_execution(struct Matrix *matrices, int num_matrices, char *operations){
  // Execute multiplication operations
  while(strchr(operations,'*')!=NULL){
    done:
    for (int i=0; operations[i]!='\0'; i++){
      if (!(operations[i]=='*')){
        continue;
      }               
      if (operations[i+1]=='\0'){
        int index_operation=i;
        // oreration index in the sting operations
        num_matrices=multiplication(matrices, num_matrices, index_operation ,index_operation+1);
        if (num_matrices==0){
          return 0;
        }
        shift_operetions(operations, index_operation); 
        goto done;
      }
      for (int j=i+1; operations[j]!='\0'; j++){
        if (operations[j]=='*'){
          continue;
        }else{
          int index_operation=j-1;
          num_matrices=multiplication(matrices, num_matrices, index_operation ,index_operation+1);
          if (num_matrices==0){
            return 0;
          }
          shift_operetions(operations, index_operation);
          goto done;
        }
      }
    }
  }
  // Execute addition and subtraction operations
  while(operations[0]!='\0'){
      int index_operation=0;
      if (operations[index_operation]=='+'){
        num_matrices=adding_up(matrices, num_matrices, index_operation ,index_operation+1);
        if (num_matrices==0){
          return 0;
        }
        shift_operetions(operations, index_operation); 
      }
      if (operations[index_operation]=='-'){
        num_matrices=subtraction(matrices, num_matrices, index_operation ,index_operation+1);
        if (num_matrices==0){
          return 0;
        }
        shift_operetions(operations, index_operation);
      }
  }
  return num_matrices;
}

// Function to shift matrices in the array to the left
int shift(struct Matrix *matrices, int num_matrices, int matrix_index){
  for (int i=matrix_index+1; i<num_matrices; i++){ 
    matrices[i-1]=matrices[i];
  }
  num_matrices--;
  return num_matrices;
}

int adding_up(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2){
  if (!((matrices[matrix_index1].rows==matrices[matrix_index2].rows)&&(matrices[matrix_index1].cols==matrices[matrix_index2].cols))){
    return 0;
  }
  for (int i=0; i<matrices[matrix_index1].len_matrix; i++){
    matrices[matrix_index1].data[i]+=matrices[matrix_index2].data[i];
  }
  free(matrices[matrix_index2].data);
  num_matrices=shift(matrices, num_matrices, matrix_index2);
  return num_matrices;
}


int subtraction(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2){
  if (!((matrices[matrix_index1].rows==matrices[matrix_index2].rows)&&(matrices[matrix_index1].cols==matrices[matrix_index2].cols))){
    return 0;
  }
  for (int i=0; i<matrices[matrix_index1].len_matrix; i++){
    matrices[matrix_index1].data[i]-=matrices[matrix_index2].data[i];
  }
  free(matrices[matrix_index2].data);
  num_matrices=shift(matrices, num_matrices, matrix_index2);
  return num_matrices;
}

int multiplication(struct Matrix *matrices, int num_matrices, int matrix_index1, int matrix_index2){
  if (!(matrices[matrix_index1].cols==matrices[matrix_index2].rows)){
    for (int i=0; i<num_matrices; i++){
      free(matrices[i].data);
    }
    return 0;
  }
  struct Matrix mult_matrix=create_matrix(matrices[matrix_index1].rows,matrices[matrix_index2].cols);
  if(mult_matrix.data==NULL){
    return 0;
  }
  for(int i=0;i<mult_matrix.rows ; i++){
    for(int j=0;j<mult_matrix.cols ; j++){
      int data_index=(i*mult_matrix.cols)+j;
      mult_matrix.data[data_index]=0;
      for (int k=0;k<matrices[matrix_index1].cols ; k++){
        mult_matrix.data[data_index]+=matrices[matrix_index1].data[(i*matrices[matrix_index1].cols)+k]*
        matrices[matrix_index2].data[j+(matrices[matrix_index2].cols)*k];
      }
    }
  }
  free(matrices[matrix_index1].data);
  matrices[matrix_index1]=mult_matrix;
  free(matrices[matrix_index2].data);
  num_matrices=shift(matrices, num_matrices, matrix_index2);
  return num_matrices;
}


struct Matrix create_matrix(int rows, int cols){
  struct Matrix matrix;
  matrix.rows=rows;
  matrix.cols=cols;
  matrix.len_matrix=rows*cols;
  matrix.data=malloc(rows*cols*sizeof(int));
  return matrix;
}



int read_matrix(struct Matrix *matrix){
  for (int i=0; i< matrix->len_matrix; i++){
    if (!((scanf("%d", &matrix->data[i]))==1)){
      return ERROR_INPUT;
    }

  }
  return OK;
}

void print_matrix(struct Matrix *matrix){
  int counter=0; 
  for (int i=0; i< matrix->len_matrix; i++){
    printf("%d",matrix->data[i]);
    if (!(matrix->cols==counter+1)){
      printf(" ");
    }
    if (matrix->cols==counter+1){ 
      printf("\n");
      counter=0;
    }else{
      counter++;
    }
  }
}



struct Matrix* get_matrices(int *num_matrices, char *operations){
  int ret;
  int allocated_matrices_capacity = 2; 
  struct Matrix *matrices = malloc((allocated_matrices_capacity)*sizeof(struct Matrix));
  int c_l_m = 0;    //count_of_loaded_matrices
  int c_o=0;          // count of operations 
  if (matrices == NULL){
    return NULL;
  }
  while (1){
    int rows, cols;
    scanf("%d %d",&rows,&cols);

    matrices[c_l_m]=create_matrix(rows, cols);
    if(matrices[c_l_m].data==NULL){
      return NULL;
    }
    ret = read_matrix(&matrices[c_l_m]);
    if (ret !=0){
      for (int i=0; i<c_l_m+1; i++){
        free(matrices[i].data);
      }
      free(matrices);
      return NULL;
    }
    c_l_m++;

    char operation='\0';
    clear_input_buffer();
    if ((scanf("%c",&operation)==1)&&((operation=='+') || (operation=='-') || (operation=='*'))){
      operations[c_o]=operation;
      c_o++;
    }else if (operation=='\0') {
      *num_matrices=c_l_m;
      operations[c_o]='\0';
      return matrices;
    }
    if (c_l_m==allocated_matrices_capacity){
    allocated_matrices_capacity*=2;
    struct Matrix * new_matrices=realloc(matrices, allocated_matrices_capacity*sizeof(struct Matrix));
      if (new_matrices == NULL){
        free(matrices);
        return NULL;
      }
      matrices=new_matrices;
    }
  }
}





