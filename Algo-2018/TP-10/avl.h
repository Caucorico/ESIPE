#ifndef AVL_H
#define AVL_H
#include "tree.h"

node *find_avl(node *t, int elt);

node *insert_avl(node *t, int elt);

node *remove_avl(node *t, int elt);

int is_avl(node *t);

#endif /* AVL_H */
