#include "tree.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

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
	if ( t == NULL )return;
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

node *find_bst(node *t, int elt)
{
	if ( t == NULL ) return NULL;
	if ( t->data == elt ) return t;
	if ( t->data > elt ) return find_bst(t->left, elt);
	else return find_bst(t->right, elt);
}

node *insert_bst(node *t, int elt)
{
	node* new_node;
	if ( t == NULL )
	{
		new_node = create_node(elt);
		return new_node;
	}

	if ( t->data == elt )
	{
		return t;
	}
	else if ( t->data < elt )
	{
		t->right = insert_bst(t->right, elt);
		return t;
	}
	else
	{
		t->left =  insert_bst(t->left, elt);
		return t;
	}
}

node *extract_min_bst(node *t, node **min)
{

	if ( t == NULL )
	{
		return NULL;
	}

	if ( t->left == NULL )
	{
		*min = t;

		return t->right;
	}

	t->left = extract_min_bst(t->left, min);
	return t;
}

node *remove_bst(node *t, int elt)
{
	node* buff;
	node* min;

	if ( t == NULL ) return NULL;

	if ( elt != t->data )
	{
		if ( elt < t->data )
		{
			t->left = remove_bst(t->left, elt);
		}
		else
		{
			t->right = remove_bst(t->right, elt);
		}
	}
	else
	{
		if ( t->left == NULL && t->right == NULL )
		{
			free(t);
			return NULL;
		}
		else if ( t->left != NULL && t->right != NULL )
		{
			t->right = extract_min_bst(t->right, &min);
			min->left = t->left;
			min->right = t->right;
			free(t);
			return min;
		}
		else
		{
			if ( t->left != NULL ) buff = t->left;
			else buff = t->right;
			free(t);
			return buff;
		}
	}

	return t;
}

node *calculate_random_insert(int n, int* nbr_elt, double* tm)
{
	int i;
	clock_t start;
	clock_t end;
	node* t = NULL;

	srand(time(NULL));

	start = clock();
	for ( i = 0 ; i < n && difftime( (start/CLOCKS_PER_SEC) + 10.0, clock()/CLOCKS_PER_SEC ) > 0.0 ; i++ )
	{
		t = insert_bst( t, rand()%(2*n) );
	}
	end = clock();

	*nbr_elt = i;
	*tm = (end/CLOCKS_PER_SEC) - (start/CLOCKS_PER_SEC);

	return t;
}

node *calculate_linear_insert(int n, int* nbr_elt, double* tm)
{
	int i;
	clock_t start;
	clock_t end;
	node* t = NULL;

	start = clock();
	for ( i = 0 ; i < n && difftime( (start/CLOCKS_PER_SEC) + 10.0, clock()/CLOCKS_PER_SEC ) > 0.0 ; i++ )
	{
		t = insert_bst( t, i );
	}
	end = clock();

	*nbr_elt = i;
	*tm = (end/CLOCKS_PER_SEC) - (start/CLOCKS_PER_SEC);

	return t;
}