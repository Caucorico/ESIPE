#include <stdio.h>
#include <stdlib.h>

typedef struct cell
{
  char* first_name;
  char* last_name;
  int age;
  struct cell* next;
}Cell, *List;
