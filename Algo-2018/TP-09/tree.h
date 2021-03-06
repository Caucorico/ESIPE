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

node *find_bst(node *t, int elt);

node *insert_bst(node *t, int elt);

node *extract_min_bst(node *t, node **min);

node *remove_bst(node *t, int elt);

node *calculate_random_insert(int n, int* nbr_elt, double* tm);

node *calculate_linear_insert(int n, int* nbr_elt, double* tm);

#endif /* TREE_H */
