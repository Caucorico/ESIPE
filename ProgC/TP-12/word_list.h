#ifndef _WORD_LIST_
#define _WORD_LIST_

typedef struct _node
{
	char* word;
	struct node* next;
}node;

char* create_word(char* w);

void free_word(char* w);

char* get_word_in_list(node* list, char* w);

char* insert_word_in_first(node* list, node* cell);

int get_list_size(node* list);

#endif

