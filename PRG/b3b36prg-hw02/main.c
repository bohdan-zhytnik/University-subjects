// Implementation of Sieve of Eratosthenes I took from this site
// https://www.geeksforgeeks.org/sieve-of-eratosthenes/


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "header.h"

// a file that will store prime numbers

#define INPUT_OK EXIT_SUCCESS
#define ERROR_INPUT 100
#define TRUE 1
#define FALSE 0
#define MAX_PRIME 1000000

void SieveOfEratosthenes(int n);
long long int read_input(void);
void prime_factorization(long long int n);
void print_factorization(int number,int *count,int *multiple);
int prime_number_control(long long int n);
int prime_check_header(long long int n,int [],int arr_len);



int main(int argc, char *argv[])
{
  int k = MAX_PRIME;
  SieveOfEratosthenes(k);
  int ret=INPUT_OK;
  long long int  n;
  while((n=read_input())>0){
    if (n==1){
      printf("Prvociselny rozklad cisla %lld je:\n",n);
      printf("%d\n",1);
    }else if(prime_number_control(n)==TRUE){
      printf("Prvociselny rozklad cisla %lld je:\n",n);
      printf("%lld\n",n);
    }else{
    printf("Prvociselny rozklad cisla %lld je:\n",n);
    prime_factorization(n);
    printf("\n");
    }
  }
  if (n<0){
    fprintf(stderr,"Error: Chybny vstup!\n");
    ret=ERROR_INPUT;
  }
  return ret;
}


long long int read_input(void){
  long long int  n = -1;
  if (scanf("%lld",&n) != 1){
    n=-1;
  }
  return n;
}

void SieveOfEratosthenes(int n)
{
    bool prime[n + 1];
    memset(prime, true, sizeof(prime));
 
    for (int p = 2; p * p <= n; p++) {
        if (prime[p] == true) {
            for (int i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
 
    int i=0;
    for (int p = 2; p <= n; p++)
        if (prime[p]){
            heade_arr[i]=p;
            i++;
        }
}

void prime_factorization(long long int n){
  int multiple = FALSE;
  int p_n_i=0;
  //arr is a list of prime numbers in some range
  //prime number index in the list arr
  //multiple means that several numbers are divisors of the entered number
  int count = 0 ;
  int * pcount=&count;
  //indicates to what power the prime number is still a divisor of an input number
  long long int n_for_loop = n;
  while(heade_arr[p_n_i] <= n_for_loop && p_n_i<heade_arr_len){
    if((n%heade_arr[p_n_i]) == 0 && n !=1){
      n=n/heade_arr[p_n_i];
      *pcount+=1;
    }else{
      print_factorization(heade_arr[p_n_i],pcount,&multiple);
      p_n_i++;
      if (n==1){
        break;
      }
    } 
  }
  p_n_i=0;
  *pcount = 0 ;
}

void print_factorization(int number,int *count,int *multiple){
  if(*multiple==TRUE){
    if (*count >0 && *count != 1 ){
      printf(" x %d^%d",number,*count);
      *count=0;
    }if (*count == 1 ) {
      printf(" x %d",number);
      *count=0;
    }
  }


  if(*multiple==FALSE){
    if (*count >0 && *count != 1 ){
      printf("%d^%d",number,*count);
      *count=0;
      *multiple = TRUE;
    }if (*count == 1 ) {
      printf("%d",number);
      *count=0;
      *multiple = TRUE;
    }
  }
}


int prime_number_control(long long int n){
  int count= prime_check_header(n, heade_arr, heade_arr_len);
  if (count==0){
    return FALSE;
  }else{
    return TRUE;  
  }
}


int prime_check_header(long long int n,int arr[],int arr_len){
    for(int i = 0; i < arr_len; i++) {
      if(arr[i] == n) {
        return TRUE;
      }
    }
    return FALSE;
}

