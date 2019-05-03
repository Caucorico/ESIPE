#include "tree.h"
#include "visualtree.h"
#include "avl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define NO_ARGUMENT INT_MIN
#define BAD_INPUT_CHAR '\0'
#define INPUT_SIZE 80

node *default_tree() {
    node *t = NULL;
    int vals[] = {7, 12, 25, 16, 21, 13, 2, 5, 4, 9, 30, 27, 6, 11, 0};
    int i;
    for (i = 0; vals[i] > 0; i++)
        t = insert_avl(t, vals[i]);
    return t;
}

void get_input(char *ch, int *val) {
    size_t len = INPUT_SIZE;
    char buffer[INPUT_SIZE+1];

    printf("Selection: ");
    fgets(buffer, len, stdin);
    
    *ch = buffer[0];
    if (sscanf(buffer+1, "%d", val) == EOF)
        *val = NO_ARGUMENT;
}

int main() {

    node *t = default_tree();
    printf("height = %d\n", height(t));
    printf("nodes = %d\n", count_nodes(t));
    write_tree(t);
    /*system("open -g -a Preview current-tree.pdf &");*/

    char ch = 'h';
    int arg = NO_ARGUMENT;
    node *tmp;
    while (ch != 'q') {
        if (arg == NO_ARGUMENT && (ch == 'i' || ch == 'e' || ch == 'f'))
            ch = BAD_INPUT_CHAR;
        switch (ch) {
        case 'f':
            tmp = find_avl(t, arg);
            if (tmp)
                printf("value %d found\n", arg);
            else
                printf("value %d not found\n", arg);
            break;
        case 'i':
            t = insert_avl(t, arg);
            write_tree(t);
            break;
        case 'r':
            t = remove_avl(t, arg);
            free(tmp);
            write_tree(t);
            break;
        case 'c':
            free_tree(t);
            t = NULL;
            write_tree(t);
            break;
        case 't':
            free_tree(t);
            t = default_tree();
            write_tree(t);
            break;
        case 'h':
            printf("+-----------------------+\n");
            printf("|         Menu          |\n");
            printf("+-----------------------+\n");
            printf("f n - find(n)\n");
            printf("i n - insert(n)\n");
            printf("r n - remove(n)\n");
            printf("c   - clear tree\n");
            printf("t   - default tree\n");
            printf("h   - display menu\n");
            printf("q   - quit\n");
            break;
        default:
            printf("Unknown selection. Use 'h' for help.\n");
        }
        get_input(&ch, &arg);
    }

    free_tree(t);

    return 0;
}
