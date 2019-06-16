#ifndef _HASH_
#define _HASH_

#define NB_PACK 4096

void init_hash_tab(node** hash_tab, int nb_pack);

unsigned long hash(char* word);

#endif