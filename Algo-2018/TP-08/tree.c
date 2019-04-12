#include "tree.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#define MAX_HEIGHT 50

node *create_node(int data) {
  node *n = (node *)malloc(sizeof(node));
  assert(n != NULL);
  n->data = data;
  n->left = NULL;
  n->right = NULL;
  return n;
}

void display_prefix(node *t)
{
	if ( t == NULL ) return;
	printf("%d ", t->data );
	display_prefix(t->left);
	display_prefix(t->right);
}

void display_infix(node *t)
{
	if ( t == NULL ) return;
	display_infix(t->left);
	printf("%d ", t->data );
	display_infix(t->right);
}

void display_suffix(node *t)
{
	if ( t == NULL ) return;
	display_suffix(t->left);
	display_suffix(t->right);
	printf("%d ", t->data );
}

node *scan_tree(void)
{
	int x;
	node* buff;

	scanf("%d", &x);
	if ( x == 0 ) return NULL;

	buff = create_node(x);
	buff->left = scan_tree();
	buff->right = scan_tree();

	return buff;
}

void free_tree(node *t)
{
	if ( t == NULL ) return;
	free_tree(t->left);
	free_tree(t->right);
	free(t);
}

void display_path(int buffer[], int index)
{
	int i;

	for ( i = 0 ; i <= index ; i++ )
	{
		printf("%d ", buffer[i]);
	}
	putchar('\n');
}

void display_paths_aux(node *t, int buffer[], int index)
{
	if ( t == NULL ) return;
	buffer[index] = t->data;
	if ( (t->left == NULL && t->right == NULL) || index >= MAX_HEIGHT )
	{
		display_path(buffer, index);
	}

	if ( index < MAX_HEIGHT)
	{
		display_paths_aux(t->left, buffer, index+1);
		display_paths_aux(t->right, buffer, index+1);
	}
}

void display_paths(node *t)
{
	int* buffer = malloc(MAX_HEIGHT*sizeof(int));

	if ( buffer == NULL )
	{
		perror("erreur malloc");
		return;
	}

	display_paths_aux(t, buffer, 0);

	free(buffer);
}