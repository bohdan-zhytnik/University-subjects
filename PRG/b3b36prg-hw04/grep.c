#include <stdio.h>
#include <stdlib.h>
// #include <regex.h>

// #define DASH '-'
// #define DASH '?'
// #define DASH '*'
// #define PLUS '+'
size_t my_strlen(char *);
int symbol_search(char *, char symbol);
int str_search(char *, char[], int len_str);
int text_search(char *, char[], int len_str);
int text_search_B(char[], char[], char[], char sign_before_r_e, int number_r_e);
void get_str_round_symbol(char[], char symbol, char *sign_before_r_e, char[], char[]);
void print_c(char *, char [], int len_str);


// int match_regex(const char *string, const char *pattern) {
//     regex_t regex;
//     int result;

//     // Компилируем регулярное выражение
//     if (regcomp(&regex, pattern, REG_EXTENDED) != 0) {
//         fprintf(stderr, "Ошибка компиляции регулярного выражения\n");
//         return 0;
//     }

//     // Проверяем совпадение регулярного выражения
//     result = regexec(&regex, string, 0, NULL, 0);
//     regfree(&regex);

//     if (result == 0) {
//         return 1; // Совпадение найдено
//     } else {
//         return 0; // Совпадение не найдено
//     }
// }

/* The main program */
int main(int argc, char *argv[])
{
  char *pattern = NULL;
  int len_pattern = 0;
  FILE *file;
  char *filename = NULL;
  int option = 0; // sign describing an optional task
  char regular_expression[4] = {'?', '*', '+'};
  int number_r_e = 0; // sign number in regular expression
  char sign_before_r_e;
  char str_before_r_e[128];
  char str_after_r_e[128];

  char E[3] = "-E";
  char clor[15] = "--color=always";
  int len_E = 2;
  int len_color = 14;

  // printf("%s\n",argv[1]);
  for (int i = 1; i < argc; i++)
  {
    // ret += symbol_search(argv[i], '-');
    if (argv[i][0] == '-')
    {
      // printf("%c\n",argv[0]);
      option += str_search(argv[i], E, len_E);
      if (str_search(argv[i], clor, len_color))
      {
        option += 2;
      }
    }
    else if (symbol_search(argv[i], '.'))
    {
      filename = argv[i];
    }
    else
    {
      for (int j = 0; j < 4; j++)
      {
        if (symbol_search(argv[i], regular_expression[j]))
        {
          if (regular_expression[j] == '?')
          {
            number_r_e = 1;
          }
          else if (regular_expression[j] == '*')
          {
            number_r_e = 2;
          }
          else
          {
            number_r_e = 3;
          }
          get_str_round_symbol(argv[i], regular_expression[j], &sign_before_r_e, str_before_r_e, str_after_r_e);
        }
      }
      pattern = argv[i];
      len_pattern = my_strlen(pattern);
    }
  }
  if (filename == NULL)
  {
    // option+=4;
    // printf("A\n");
  }
  // printf("\033[0;31mHello, hello\033[0m\n");
  // printf("\033[01;31mHello, hello\033[K\033[m\n");
  // printf("\033[01;31m\033[KHello, hello\033[m\033[K\n");
  // printf("H\033[01;31m\033[Kel\033[m\033[Klo, ");
  // printf("h\033[01;31m\033[Kel\033[m\033[Klo\n");

  // printf("len_pattern  %d\n",len_pattern);
  // printf("str_before_r_e:   %s\n", str_before_r_e);
  // printf("sign_before_r_e  %c\n",sign_before_r_e);
  // printf("str_after_r_e:   %s\n", str_after_r_e);

  // printf("number_r_e  %d\n",number_r_e);
  // printf("filename:   %s\n", filename);
  // printf("pattern:   %s\n", pattern);
  // printf("%d\n", option);
  if (filename != NULL)
  {
    file = fopen(filename, "r");

  }
  else
  {
    file = stdin;
  }
  if (file == NULL)
  {
    fprintf(stderr, "Не удалось открыть файл %s\n", filename);
    return 1;
  }
  int control = number_r_e;
  control = 1;

  char line[128];
  while (fgets(line, sizeof(line), file) != NULL)
  {
    // printf("WOW");
    // printf("if");
    if (option == 0)
    {
      if (text_search(line, pattern, len_pattern))
      {
        printf("%s", line);
        control = 0;
      }
    }
    else if (option == 1)
    {
    if (match_regex(line, pattern)) {
        printf("Совпадение найдено\n");
    } else {
        printf("Совпадение не найдено\n");
    }
      if (text_search_B(line, str_before_r_e, str_after_r_e, sign_before_r_e, number_r_e) == 1)
      {
        printf("%s", line);
        // printf("WOW\n");
        control = 0;
      }
    }
    else if (option == 2){
      // printf("wow");
      if (text_search(line, pattern, len_pattern)){
        print_c(line, pattern, len_pattern);
        control = 0;
        // printf("\n");  
      }
    }
  }
  // Закрываем файл
  // printf("\n");
  fclose(file);

  return control;
}

void print_c(char *argv, char str[], int len_str){
  for (int i=0; argv[i]!='\0';i++){
    int count=0;
    if (argv[i]==str[0]){
      int i_for_loop=i;
      for (int j=0; str[j]!='\0'; j++){
        if (argv[i_for_loop]==str[j]){
          i_for_loop++;
          count++;
        }else{
          count=0;
          break;
        }
        if (count==len_str){
          // printf("\033[01;31m\033[Kstr\033[m\033[K");
          // char tmp[len_str+1];
          
          // for (int k = 0; str[k]!=0; k++){
          // ;
            printf("\033[01;31m\033[K%s\033[m\033[K",str);
          // }
          i=i+len_str-1;
        }

      }
    }else{
      printf("%c",argv[i]);
    }
  }


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

int symbol_search(char *argv, char symbol)
{
  for (int i = 0; argv[i] != '\0'; i++)
  {
    if (argv[i] == symbol)
    {
      return 1;
    }
  }
  return 0;
}

int str_search(char *argv, char str[], int len_str)
{
  int count = 0;
  for (int i = 0; str[i] != '\0'; i++)
  {
    if (argv[i] == str[i])
    {
      count++;
    }
    if (count == len_str)
    {
      return 1;
    }
  }
  return 0;
}

int text_search(char *argv, char str[], int len_str)
{
  int count = 0;
  for (int i = 0; argv[i] != '\0'; i++)
  {
    if (argv[i] == str[count])
    {
      count++;
      if (count == len_str)
      {
        return 1;
      }
    }
    else
    {
      count = 0;
    }
  }
  return 0;
}

int text_search_B(char *argv, char *str_before_r_e, char *str_after_r_e, char sign_before_r_e, int number_r_e)
{
  // printf("argv:   %s", argv);
  int control_b = 0;
  int control_s = 0;
  // int control_a=0;
  int control = 0;
  int count = 0;
  int len_before = 0;
  int len_after = 0;
  int my_len_b = my_strlen(str_before_r_e);
  int my_len_a = my_strlen(str_after_r_e);
  int c = 0;
  for (int i = 0; argv[i] != '\0'; i++)
  {
    // printf("my_len_b %d\n",my_len_b);
    if (my_len_b == 0 && c == 0)
    {
      control_b = 1;
      c = 1;
      // printf("c1");
    }
    if (control_s == 1)
    {
      // printf("my_len_a %d\n",my_len_a);

      // if (str_after_r_e[0]=='\0'){
      //   printf("wtf");
      // }else{
      // printf("lol");}
      // printf("wtf%c\n",str_after_r_e[1]);
      if (my_len_a == 0)
      {
        control = 1;
        control_b = -1;
        return 1;
      }
      control_b = -1;
      // c=-1;
      // len_after=0;
      // printf("a\n");
      // count=0;
      // printf("ai%c\n",argv[i]);
      // printf("sa%c\n",str_after_r_e[len_after]);
      if (argv[i] == str_after_r_e[len_after])
      {
        // printf("aa\n");
        len_after++;
        count++;
        if (count == my_len_a)
        {
          control = 1;
          return 1;
          // printf("yes\n");
        }
      }
      else
      {
        control_b = 0;
        control_s = 0;
        len_after = 0;
        count = 0;
        // c=0;
      }
    }
    if (control_b == 1)
    {
      // printf("%c\n",argv[i]);
      // if (my_len_b==0 && c==0){
      //   // printf("%d\n",i);
      //   i=i-1;
      //   c=1;
      // }
      // printf("start contrlor s\n");
      if (number_r_e == 1)
      {
        // printf("%c\n",argv[i]);
        if (!(argv[i] == sign_before_r_e))
        {
          // printf("no\n");

          if (argv[i] == str_after_r_e[len_after])
          {
            // printf("yes\n");
            len_after++;
            count++;
            // printf("lenA%d\n",my_len_a);
            // printf("%d\n",count);
            if (count == my_len_a)
            {
              control = 1;
              return 1;
              // printf("yes\n");
            }
          }
          else
          {
            // c=0;
            control_b = 0;
            // control_s=0;
            len_after = 0;
            count = 0;
          }
        }
        else
        {
          control_s = 1;
          control_b = -1;
          // printf("yes\n");
        }
      }

      if (number_r_e == 2)
      {
        // if (my_len_b==0 && c==0){
        //   // printf("%d\n",i);
        //   i=i-1;
        //   // c=1;
        // }
        if (argv[i] == sign_before_r_e)
        {
          count++;
          // control_s=1;
          // printf("s\n");
        }
        else if (count != 0)
        {
          // printf("count!=0\n");
          i = i - 1;
          control_s = 1;
          control_b = -1;
          count = 0;
        }
        else
        {
          i = i - 1;
          // printf("else\n");
          control_b = -1;
          control_s = 1;
          count = 0;
        }
      }

      if (number_r_e == 3)
      {
        // if (my_len_b==0 && c==0){
        //   // printf("%d\n",i);
        //   i=i-1;
        //   c=1;
        // }
        // printf("start contrlor S\n");
        // printf("%c\n",argv[i]);
        // printf("%c\n",sign_before_r_e);
        if (argv[i] == sign_before_r_e)
        {
          count++;
          // control_s=1;
          // printf("s\n");
        }
        else if (count != 0)
        {
          // printf("count!=0\n");
          i = i - 1;
          control_s = 1;
          control_b = -1;
          count = 0;
        }
        else
        {
          // printf("else\n");
          control_b = 0;
          count = 0;
        }
      }
    }
    if (control_b == 0)
    {
      if (argv[i] == str_before_r_e[len_before])
      {
        len_before++;
        count++;
        if (count == my_len_b)
        {
          control_b = 1;
          count = 0;
          // printf("b\n");
        }
      }
      else
      {
        count = 0;
        // c=0;
      }
    }
    // if (control_b==1){
    //   if ((argv[i]==sign_before_r_e)){
    //     control_s=1;
    //     printf("s\n");
    //   }else{
    //     control_b=0;
    //     count=0;
    //   }
    // }
    // if(control_s==1){
    //   count=0;
    //   if ((argv[i]==str_after_r_e[len_after])){
    //       len_after++;
    //       count++;
    //       if (count==my_strlen(str_after_r_e)){
    //         control=1;
    //         printf("yes\n");
    //       }
    //   }else{
    //     control_b=0;
    //     control_s=0;
    //   }
    // }
    if (control == 1)
    {
      return 1;
    }
  }
  return 0;
}

void get_str_round_symbol(char *argv, char symbol, char *sign, char *str, char *str_after)
{
  char front_char;
  int control = 0;
  int len_str_after = 0;
  for (int i = 0; argv[i] != '\0'; i++)
  {
    if (control == 1)
    {
      str_after[len_str_after] = argv[i];
      len_str_after++;
    }
    if (argv[i] != symbol)
    {
      if (i != 0)
      {
        str[i - 1] = front_char;
      }
      front_char = argv[i];
    }
    if (argv[i] == symbol)
    {
      *sign = front_char;
      str[i-1]='\0';
      control = 1;
    }
  }
  str_after[len_str_after] = '\0';
}


