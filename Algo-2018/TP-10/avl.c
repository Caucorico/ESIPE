#include "tree.h"
#include "avl.h"
#include <stddef.h> /* NULL */
#include <stdlib.h>
#include <stdio.h>

int max(int a, int b)
{
    if ( a > b )return a;
    return b;
}

/* SEARCH */

node *find_avl(node *t, int elt) {
    return NULL;
}

/* UPDATE HEIGHTS */

void update_height(node *t)
{
    if ( t->left == NULL && t->right == NULL)
    {
        t->height = 0;
    }
    else if ( t->left == NULL )
    {
        t->height = 1+t->right->height;
    }
    else if ( t->right == NULL )
    {
        t->height = 1+t->left->height;
    }
    else
    {
        t->height = 1+max(t->left->height, t->right->height);
    }
}

/* ROTATIONS */

/*
 *     r            c
 *    / \          / \
 *   c   C   =>   A   r
 *  / \              / \
 * A   B            B   C
 */
node *rotate_right(node *y)
{
    node* c;

    if ( y == NULL ) return NULL;   
    if ( y->left == NULL ) return y;

    c = y->left;
    y->left = c->right;
    c->right = y;

    update_height(y);
    update_height(c);

    return c;
}

/*
 *     r            c
 *    / \          / \
 *   A   c   =>   r   C
 *      / \      / \
 *     B   C    A   B
 */
node *rotate_left(node *x)
{
    node* c;

    if ( x == NULL ) return NULL;
    if ( x->right == NULL ) return x;

    c = x->right;
    x->right = c->left;
    c->left = x;

    update_height(x);
    update_height(c);

    return c;
}

node *rotate_left_right(node *y)
{
    y->left = rotate_left(y->left);
    y = rotate_right(y);
    return y;
}

node *rotate_right_left(node *x)
{
    x->right = rotate_right(x->right);
    x = rotate_left(x);
    return x;
}

/* REBALANCE */

int compute_balance(node *t)
{
    if ( t == NULL ) return 0;
    else if ( t->left == NULL && t->right == NULL ) return 0;
    else if ( t->left == NULL ) return -t->right->height;
    else if ( t->right == NULL ) return t->left->height;
    return t->left->height - t->right->height;
}

node *rebalance(node *t)
{
    int eq, eq2;

    if ( t == NULL ) return NULL;

    eq = compute_balance(t);
    if ( eq > 1 )
    {
        eq2 = compute_balance(t->left);
        if ( eq2 >= -1 )
        {
            t = rotate_left_right(t);
        }
        else
        {
            t = rotate_right(t);
        }
    }
    else if ( eq < -1 )
    {
        eq2 = compute_balance(t->right);
        if ( eq2 >= 0 )
        {
            t = rotate_right_left(t);
        }
        else
        {
            t = rotate_left(t);
        }
    }

    return t;
}

/* INSERTION */

node *insert_avl(node *t, int elt)
{
    if ( t == NULL )
    {
        return create_node(elt);
    }

    if ( elt < t->data )
    {
        t->left = insert_avl(t->left, elt);
    }
    else
    {
        t->right = insert_avl(t->right, elt);
    }

    update_height(t);
    t = rebalance(t);

    return t;
}

/* CHECK */

int is_avl(node *t)
{
    return 0;
}

/* REMOVAL */

node *extract_min_avl(node *t, node **min) {
    return NULL;
}

node *remove_avl(node *t, int elt) {
    return NULL;
}
