#include "tree.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

node *create_node(int elt) {
  node *n = (node *)malloc(sizeof(node));
  if (n != NULL) {
    n->data = elt;
    n->height = 0;
    n->left = NULL;
    n->right = NULL;
  }
  return n;
}

void display_prefix(node *t) {
  if (t != NULL) {
    printf("%d ", t->data);
    display_prefix(t->left);
    display_prefix(t->right);
  }
}

void display_infix(node *t) {
  if (t != NULL) {
    display_infix(t->left);
    printf("%d ", t->data);
    display_infix(t->right);
  }
}

void display_suffix(node *t) {
  if (t != NULL) {
    display_suffix(t->left);
    display_suffix(t->right);
    printf("%d ", t->data);
  }
}

node *scan_tree(void) {
  int val;
  scanf("%d",&val);
  if (val == 0) {
    return NULL;
  }
  node *n = create_node(val);
  n->left = scan_tree();
  n->right = scan_tree();
  return n;
}

int MAX(int a, int b) {
  return a > b ? a : b;
}

int height(node *t) {
  if (t == NULL)
    return -1;
  else
    return 1+MAX(height(t->left),height(t->right));
}

int count_nodes(node *t) {
  if (t == NULL)
    return 0;
  else
    return 1+count_nodes(t->left)+count_nodes(t->right);
}

int count_internal_nodes(node *t) {
  if (t == NULL)
    return 0;
  else if (t->left == NULL && t->right == NULL)
    return 0;
  else
    return 1+count_internal_nodes(t->left)+count_internal_nodes(t->right);
}

int count_leaves(node *t) {
  if (t == NULL)
    return 0;
  else if (t->left == NULL && t->right == NULL)
    return 1;
  else
    return count_leaves(t->left)+count_leaves(t->right);
}

int count_full_nodes(node *t) {
  int sum;
  if (t == NULL)
    return 0;
  sum = count_full_nodes(t->left)+count_full_nodes(t->right);
  if (t->left != NULL && t->right != NULL)
    sum++;
  return sum;
}

int sum(node *t) {
  if (t == NULL)
    return 0;
  return t->data+sum(t->left)+sum(t->right);
}

int sum_depth_aux(node *t, int depth) {
  if (t == NULL)
    return 0;
  return depth+
      sum_depth_aux(t->left, depth+1)+
      sum_depth_aux(t->right, depth+1);
}

int sum_depth(node *t) {
  return sum_depth_aux(t, 0);
}

void free_tree(node *t) {
  if (t != NULL) {
    free_tree(t->left);
    free_tree(t->right);
    free(t);
  }
}

void display_paths_aux(node *t, int buffer[], int index) {
  int i;
  if (t == NULL)
    return;
  buffer[index] = t->data;
  if (t->left == NULL && t->right == NULL) {
    for (i = 0; i <= index; i++)
      printf("%d ", buffer[i]);
    printf("\n");
  }
  else {
    display_paths_aux(t->left, buffer, index+1);
    display_paths_aux(t->right, buffer, index+1);
  }
}

void display_paths(node *t) {
  int *buffer = (int *)malloc(MAX_HEIGHT*sizeof(int));
  display_paths_aux(t, buffer, 0);
  free(buffer);
}
