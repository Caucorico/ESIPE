#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "menu.h"
#include "visualtree.h"

void display_possibility(void)
{
	printf("h : help \n");
	printf("s <nodes> : create a new tree with given nodes \n");
	printf("a <nb_nodes> : create a random tree with nb_nodes \n");
	printf("c <nb_nodes> : create a random tree with nb_nodes and give infos \n");
	printf("l <nb_nodes> : create a linear tree with nb_nodes and give infos \n");
	printf("i <element> : insert the element \n");
	printf("f <element> : search the element \n");
	printf("d : display the sorted tree \n");
	printf("m : extract min \n");
	printf("r <element> : remove the element \n");
	printf("t : terminate \n");
}

node* create_new_tree( void )
{
	return scan_tree();
}

node* create_random_tree( int n )
{
	int i;
	node* new_tree = NULL;

	srand(time(NULL));

	for ( i = 0 ; i < n ; i++ )
	{
		new_tree = insert_bst( new_tree, rand()%0xfff );
	}

	return new_tree;
}

node* insert_element( node* t, int n )
{
	return insert_bst( t, n );
}

void search_element( node* t, int n )
{
	if ( find_bst( t, n ) != NULL )
	{
		printf("found ! \n");
	}
	else
	{
		printf("not found ! \n");
	}
}

void display_sorted_tree( node* t )
{
	display_infix(t);
}

node* extract_min( node* t )
{
	node* min;

	if ( t == NULL )
	{
		printf("The tree is empty ! \n");
	}
	else
	{
		t = extract_min_bst( t, &min );
		printf("min : %d", min->data);
	}

	return t;
}

node* remove_el( node* t, int elt )
{
	t = remove_bst( t, elt );
	return t;
}

void test_random( int n )
{
	int nbr_elt;
	double tm;
	node* t;
	t = calculate_random_insert(n, &nbr_elt, &tm);
	free_tree(t);
	printf("results : %d elements created in %lf seconds \n", nbr_elt, tm);
}

void test_linear( int n )
{
	int nbr_elt;
	double tm;
	node* t;
	t = calculate_random_insert(n, &nbr_elt, &tm);
	free_tree(t);
	printf("results : %d elements created in %lf seconds \n", nbr_elt, tm);
}

void manage_command(void)
{
	char c;
	int b;
	node* tree = NULL;

	do
	{
		printf("=>");
		scanf(" %c", &c);

		switch (c)
		{
			case 'h':
				display_possibility();
				break;

			case 's':
				free_tree(tree);
				tree = scan_tree();
				break;

			case 'a':
				free_tree(tree);
				scanf("%d", &b);
				tree = create_random_tree(b);
				break;

			case 'i':
				scanf("%d", &b);
				tree = insert_element(tree, b);
				break;

			case 'f':
				scanf("%d", &b);
				search_element(tree, b);
				break;

			case 'd':
				display_sorted_tree( tree );
				putchar('\n');
				break;

			case 'm':
				tree = extract_min( tree );
				putchar('\n');
				break;

			case 'r':
				scanf("%d", &b);
				tree = remove_el( tree, b );
				putchar('\n');
				break;

			case 'c':
				scanf("%d", &b);
				test_random(b);
				break;

			case 'l':
				scanf("%d", &b);
				test_linear(b);
				break;

			case 't':
				break;

			default:
				printf("Command not found !\n");

		}
		write_tree( tree );
	} while ( c!= 't' );

	free_tree(tree);
}