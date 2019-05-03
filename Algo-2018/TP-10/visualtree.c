#include "visualtree.h"
#include <stdio.h>
#include <stdlib.h>

FILE* write_begin(char *name) {
    FILE *f = fopen(name, "w");
    fprintf(f, "digraph tree {\n");
    fprintf(f, "  node [shape=record,height=.1]\n");
    fprintf(f, "  edge [tailclip=false, arrowtail=dot, dir=both];\n\n");
    return f;
}

void write_end(FILE *f) {
    fprintf(f, "\n}\n");
    fclose(f);
}

void write_node(FILE *f, node *n) {
    fprintf(f, "  n%p [label=\"<left> |{ <value> %d | <height> %d }| <right>\"];\n", (void *)n, n->data, n->height);
}

void write_left_link(FILE *f, node *n) {
    fprintf(f, "  n%p:left:c -> n%p:value;\n", (void *)n, n->left);
}

void write_right_link(FILE *f, node *n) {
    fprintf(f, "  n%p:right:c -> n%p:value;\n", (void *)n, n->right);
}

void write_tree_aux(FILE *f, node *t) {
    if (t == NULL)
        return;
    write_node(f, t);
    if (t->left != NULL) {
        write_left_link(f, t);
        write_tree_aux(f, t->left);
    }
    if (t->right != NULL) {
        write_right_link(f, t);
        write_tree_aux(f, t->right);
    }
}

void write_tree(node *t) {
  FILE *f;
  f = write_begin("current-tree.dot");
  write_tree_aux(f, t);
  write_end(f);
  system("dot -Tps2 current-tree.dot | ps2pdf - - > current-tree.pdf");
}
