/* test-td4-ex2.c */
/* la fonction plus(int,int) renvoie
   la somme de ses paramètres */
int plus
   (int a,int b) {
   return a+b;
}
int main (void) {
   printf ("plus(4,7)=%d\n",plus(4,7));
   getchar(); // getchar() attend un retour chariot
   return 0;
}