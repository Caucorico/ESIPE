#include <stdio.h>

void print_var( int n )
{
  printf("Value of the variable : %d\n", n);
}

void print_pointer( int* p )
{
  printf("Pointer address %p and Pointed Value : %d\n", p, *p );
}

void set_pointer( int* p, int n )
{
  *p = n;
}

int main( int argc, char *argv[] )
{
  int a;
  int* p=&a;

  printf("########################################\n");
  printf("TP-01 Exercice-04. \nBut : Comprendre les pointeurs. \n\n");

  print_var(a);
  a = 53;
  print_var(a);
  print_pointer(p);
  set_pointer(p, 42);
  print_pointer(p);
  print_var(a);

  printf("\n\n########################################\n");

  return 0;
}