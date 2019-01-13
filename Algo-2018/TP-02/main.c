#include <stdio.h>
#include "ex1.h"
#include "ex2.h"
#include "ex3.h"
#include "ex4.h"
#include "ex5.h"
#include "ex6.h"

int main(void)
{

  char t1[20] = { 'a', 'b', 'a', '\0' };
  char t2[20] = { 'a', 'b', 'b', '\0' };
  char t3[20] = { 'a', '\0' };
  int t4[3] = { 1, 1, 2 };
  int t5[3] = { 1, 1, 1 };
  int t6[3] = { 1, 2, 3 };
  int t7[8] = { 1, 2, 3, 2, 4, 6, 8, 3 };
  int t8[8] = { 1, 2, 3, 1, 2, 3, 4, 5 };

  printf("############## EXERCICE 01 ##############\n\n");

  printf("Test de la fonction << palindrome_rec >>\n\n");

  printf("Test 1 : Envoyer { 'a', 'b', 'a' }.\n");

  printf("Réponse attendue : vrai.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome_rec(t1, 0, 2));

  printf("Test 2 : Envoyer { 'a', 'b', 'b' }.\n");

  printf("Réponse attendue : faux.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome_rec(t2, 0, 2));

  printf("Test 3 : Envoyer { 'a' }.\n");

  printf("Réponse attendue : vrai.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome_rec(t3, 0, 0));

  printf("Test de la fonction << palindrome >>\n\n");

  printf("Test 1 : Envoyer { 'a', 'b', 'a' }.\n");
  printf("Réponse attendue : vrai.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome(t1));

  printf("Test 2 : Envoyer { 'a', 'b', 'b' }.\n");
  printf("Réponse attendue : faux.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome(t2));

  printf("Test 3 : Envoyer { 'a' }.\n");
  printf("Réponse attendue : vrai.\n");
  printf("Réponse obtenue : %d.\n\n", palindrome(t3));

  printf("############## EXERCICE 02 ##############\n\n");

  printf("Test de la fonction << increasing_sequence_rec >>\n\n");

  printf("Test 1 : n=0.\n");
  printf("Réponse attendue : rien.\n");
  printf("Réponse obtenue : \n");
  increasing_sequence_rec(0);
  putchar('\n');
  putchar('\n');

  printf("Test 2 : n=5.\n");
  printf("Réponse attendue : une suite croissante de 1 a 5.\n");
  printf("Réponse obtenue :\n");
  increasing_sequence_rec(5);
  putchar('\n');
  putchar('\n');

  printf("Test 3 : n=10.\n");
  printf("Réponse attendue : une suite croissante de 1 a 10.\n");
  printf("Réponse obtenue : \n");
  increasing_sequence_rec(10);
  putchar('\n');
  putchar('\n');

  printf("Test de la fonction << decreasing_sequence_rec >>\n\n");

  printf("Test 1 : n=0.\n");
  printf("Réponse attendue : rien.\n");
  printf("Réponse obtenue : \n");
  decreasing_sequence_rec(0);
  putchar('\n');
  putchar('\n');

  printf("Test 2 : n=5.\n");
  printf("Réponse attendue : une suite decroissante de 5 a 1.\n");
  printf("Réponse obtenue :\n");
  decreasing_sequence_rec(5);
  putchar('\n');
  putchar('\n');

  printf("Test 3 : n=10.\n");
  printf("Réponse attendue : une suite decroissante de 10 a 1.\n");
  printf("Réponse obtenue : \n");
  decreasing_sequence_rec(10);
  putchar('\n');
  putchar('\n');

  printf("############## EXERCICE 03 ##############\n\n");

  printf("Test de la fonction << count >>\n\n");

  printf("Test 1 : Envoyer { 1, 1, 2 } et elt = 1.\n");
  printf("Réponse attendue : 2.\n");
  printf("Réponse obtenue : %d.\n\n", count(t4, 0, 2, 1));

  printf("Test 2 : Envoyer { 1, 1, 2 } et elt = 2.\n");
  printf("Réponse attendue : 1.\n");
  printf("Réponse obtenue : %d.\n\n", count(t4, 0, 2, 2));

  printf("Test 3 : Envoyer { 1, 1, 2 } et elt = 3.\n");
  printf("Réponse attendue : 0.\n");
  printf("Réponse obtenue : %d.\n\n", count(t4, 0, 2, 3));

  printf("Test de la fonction << max_count >>\n\n");

  printf("Test 1 : Envoyer { 1, 1, 2 }.\n");
  printf("Réponse attendue : 2.\n");
  printf("Réponse obtenue : %d.\n\n", max_count(t4, 0, 2));

  printf("Test 2 : Envoyer { 1, 1, 1 }.\n");
  printf("Réponse attendue : 3.\n");
  printf("Réponse obtenue : %d.\n\n", max_count(t5, 0, 2));

  printf("Test 3 : Envoyer { 1, 2, 3 }.\n");
  printf("Réponse attendue : 1.\n");
  printf("Réponse obtenue : %d.\n\n", max_count(t6, 0, 2));

  printf("############## EXERCICE 04 ##############\n\n");

  printf("Test de la fonction << sum_digits_iter >>\n\n");

  printf("Test 1 : Envoyer 912942.\n");
  printf("Réponse attendue : 27.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_iter(912942));

  printf("Test 2 : Envoyer 666.\n");
  printf("Réponse attendue : 18.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_iter(666));

  printf("Test 3 : Envoyer 111111111.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_iter(9));

  printf("Test de la fonction << sum_digits_rec >>\n\n");

  printf("Test 1 : Envoyer 912942.\n");
  printf("Réponse attendue : 27.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_rec(912942));

  printf("Test 2 : Envoyer 666.\n");
  printf("Réponse attendue : 18.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_rec(666));

  printf("Test 3 : Envoyer 111111111.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", sum_digits_rec(9));

  printf("############## EXERCICE 05 ##############\n");
  printf("###### NE RESPECTE PAS LA CONSIGNE ######\n\n");

  printf("Test de la fonction << digit_sum_digits_iter >>\n\n");

  printf("Test 1 : Envoyer 912942.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_iter(912942));

  printf("Test 2 : Envoyer 666.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_iter(666));

  printf("Test 3 : Envoyer 111111111.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_iter(9));

  printf("Test de la fonction << digit_sum_digits_rec >>\n\n");

  printf("Test 1 : Envoyer 912942.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_rec(912942));

  printf("Test 2 : Envoyer 666.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_rec(666));

  printf("Test 3 : Envoyer 111111111.\n");
  printf("Réponse attendue : 9.\n");
  printf("Réponse obtenue : %d.\n\n", digit_sum_digits_rec(9));

  printf("############## EXERCICE 06 ##############\n");

  printf("Test de la fonction << longest_incr_iter >>\n\n");

  printf("Test 1 : Envoyer { 1, 2, 3, 2, 4, 6, 8, 3}.\n");
  printf("Réponse attendue : 4.\n");
  printf("Réponse obtenue : %d.\n\n", longest_incr_iter(t7, 0, 7));

  printf("Test 2 : Envoyer { 1, 2, 3, 1, 2, 3, 4, 5 }.\n");
  printf("Réponse attendue : 5.\n");
  printf("Réponse obtenue : %d.\n\n", longest_incr_iter(t8, 0, 7));


  return 0;
}