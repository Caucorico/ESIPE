#include <stdio.h> 
#include <stdlib.h>

int pascal (int nBut, int pBut){
 int * tab;
 unsigned int n, i;

 tab = (int *)malloc ((nBut+1)*sizeof(int));
 if(tab==NULL){
   fprintf(stderr,"Pas assez de place\n");
   exit(0);
 }

 tab[0] = 1;

 for(n=1; n<=nBut; n++){
   tab[n] = 1;

   for(i=n-1; i>0; i--)
     tab[i] = tab[i-1] + tab[i];
 }

 int result=tab[pBut];
 free(tab);
 return result;
}

int main(int argc, char * argv[]) {
   printf(" Cn, p = %d\n", pascal (30000, 250));
   return 0;
}
