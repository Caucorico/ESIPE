#include "tree.h"
#include <stdio.h>
#include "visualtree.h"

int main() {

  /*node *trois = create_node(3);
  node *deux = create_node(2);
  node *cinq = create_node(5);
  node* douze = create_node(12);
  node *un = create_node(1);
  node* quatre = create_node(4);
  node* sept = create_node(7);
  trois->left = cinq;
  trois->right = deux;
  cinq->left = douze;
  cinq->right = un;
  douze->left = NULL;
  douze->right = NULL;
  un->left = quatre;
  un->right = NULL;
  quatre->left = NULL;
  quatre->right = NULL;
  deux->left = NULL;
  deux->right = sept;
  sept->left = NULL;
  sept->right = NULL;

  write_tree(trois); */

  node* test;

  test = scan_tree();

  write_tree(test);

  display_paths(test);

  free_tree(test);


  return 0;
}
