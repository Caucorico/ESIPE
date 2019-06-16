#include <stdlib.h>
#include "word_list.h"
#include "hash.h"

void init_hash_tab(node** hash_tab, int nb_pack)
{
	int i;

	for ( i = 0 ; i < nb_pack ; i++ )
	{
		hash_tab[i] = NULL;
	}
}

unsigned long hash(char* word)
{
	int i;
	unsigned long h;

	h = 0;

	for ( i = 0 ; word[i] != '\0' ; i++ )
	{
		h += (word[i]*(i+1));
	}

	return h%NB_PACK;
}