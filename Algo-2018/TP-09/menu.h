#ifndef _MENU_
#define _MENU_

#include "tree.h"

/* Display the commands possibility */
void display_possibility(void);

/* Create a new tree with the number in stdin */
node* create_new_tree( void );

/* Create a random tree with n elements */
node* create_random_tree( int n );

/* Insert the n element in the tree t */
node* insert_element( node* t, int n );

/* Search the n element in the tree t */
void search_element( node* t, int n );

/* Display the tree in infix order */
void display_sorted_tree( node* t );

/* Manage the command input */
void manage_command(void);

#endif