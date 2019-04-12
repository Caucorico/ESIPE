#include "visualtree.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Open a file and start writing the DOT code for a tree.
 * Returns a pointer to the file.
 */
FILE* write_begin(char *name) {
  FILE *f = fopen(name, "w");
  fprintf(f, "digraph tree {\n");
  fprintf(f, "  splines=false\n");
  fprintf(f, "  node [shape=record,height=.1]\n");
  fprintf(f, "  edge [tailclip=false, arrowtail=dot, dir=both];\n\n");
  return f;
}

/*
 * Write the terminating brace and close the file.
 */
void write_end(FILE *f) {
  fprintf(f, "\n}\n");
  fclose(f);
}

/*
 * Write the DOT code for a single node n to an open file f.
 */
void write_node(FILE *f, node *n) {
  fprintf(f, "  n%p [label=\"<left> | <value> %d | <right>\"];\n", n, n->data);
}

/*
 * Write the DOT code declaring a left child of node n to an open file f.
 */
void write_left_link(FILE *f, node *n) {
  fprintf(f, "  n%p:left:c -> n%p:value;\n", n, n->left);
}

/*
 * Write the DOT code declaring a right child of node n to an open file f.
 */
void write_right_link(FILE *f, node *n) {
  fprintf(f, "  n%p:right:c -> n%p:value;\n", n, n->right);
}

/*********************************************************/
/*********************************************************/
/*********************************************************/
/*
 * This function currently ignores the tree t and outputs a 
 * hard-coded tree instead. REPLACE THE ENTIRE FUNCTION!
*/
void write_tree_aux(FILE *f, node *t) {

  if ( t == NULL ) return;

  write_node(f,t);
  if ( t->left != NULL )
    write_left_link(f,t);

  if ( t->right != NULL )
    write_right_link(f,t);

  write_tree_aux(f, t->left);
  write_tree_aux(f, t->right);

  /* **** */

}
/*********************************************************/
/*********************************************************/
/*********************************************************/

/*
 * Open a file current-tree.dot, write the DOT code for a tree t, 
 * and convert the .dot-file to a pdf.
 */
void write_tree(node *t) {
  FILE *f;
  f = write_begin("current-tree.dot");
  write_tree_aux(f, t);
  write_end(f);
  system("dot -Tps2 current-tree.dot | ps2pdf - - > current-tree.pdf");
}