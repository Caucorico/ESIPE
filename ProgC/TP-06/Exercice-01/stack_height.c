#include <stdio.h>
#include <stdlib.h>

/* Les gros tableau doivent etre mis la. La pile est trop petite. */
int big_tab[10000000];

/* Display indentation following the depth of recursion */
void make_space(int n){
  int i;

  for (i=0 ; i<n ; i++)
    printf("  ");
}

/* Recursive function displaying the height of the execution stack */
int stack_adr(int n, int* height, int* start_adr){
  int p=n;

  /* Ca sa fait plante */
  int big_tab2[10000000];

  if (n >= 0){
    make_space(n);
    printf("Recursive climbing call : %d\n", p);
    make_space(n);
    printf("Adr pointer : %p and Adr local variable : %p\n", height, &p);
    make_space(n);
    printf("Stack jump : %ld\n", height-&p);
    make_space(n);
    printf("Stack total height : %ld\n\n", start_adr-&p);

    stack_adr(n-1, &p, start_adr);

    make_space(n);
    printf("Recursive descending call : %d\n", p);
    make_space(n);
    printf("Stack total height : %ld\n\n", start_adr-&p);
  }
  else{
    printf("LAST RECURSION\n\n");
  }
  return p;
}

int main(int argc, char* argv[]){
  int i;

  if (argc != 2){
    fprintf(stderr, "Nope, I need one positive integer in argument to work properly.\n");
  }

  stack_adr(atoi(argv[1]), &i, &i);

  return 0;
}
