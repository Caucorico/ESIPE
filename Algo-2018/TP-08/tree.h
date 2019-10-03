#ifndef TREE_H
#define TREE_H

typedef struct _node {
    int data;                /* data stored : an integer    */
    struct _node *left;      /* pointer to the left child   */
    struct _node *right;     /* pointer to the right child  */
} node;

/*
 * Allocate memory for a new node.
 */
node *create_node(int data);

void display_prefix(node *t);

void display_infix(node *t);

void display_suffix(node *t);

node *scan_tree(void);

void free_tree(node *t);

void display_paths(node *t);

#endif /* TREE_H */
