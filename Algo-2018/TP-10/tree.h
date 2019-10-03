#ifndef TREE_H
#define TREE_H

#define MAX_HEIGHT 200

typedef struct _node {
    int data;                /* donnee stockee : un entier  */
    int height;              /* la hauteur de l'arbre       */
    struct _node *left;      /* pointeur sur le fils gauche */
    struct _node *right;     /* pointeur sur le fils droit  */
} node;

node *create_node(int elt);

void display_prefix(node *t);
void display_infix(node *t);
void display_suffix(node *t);
node *scan_tree(void);

int height(node *t);
int count_nodes(node *t);
int count_internal_nodes(node *t);
int count_leaves(node *t);
int count_full_nodes(node *t);
int sum_depth(node *t);

void free_tree(node *t);

void display_paths(node *t);

#endif /* TREE_H */
