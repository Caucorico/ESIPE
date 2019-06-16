#ifndef _WORD_LIST_
#define _WORD_LIST_

typedef struct _node
{
	char* word;
	struct _node* next;
}node;

char* create_word(char* w);

node* create_word_node(char* w);

void free_word(char* w);

node* get_word_in_list(node* list, char* w);

int insert_word_in_first(node** list, char* w);

int get_list_size(node* list);

int get_hash_tab_size(node** hash_tab);

#endif

