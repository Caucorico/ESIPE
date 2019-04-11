#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

olink *create_olink(int pos) {
    olink *olnk = malloc(sizeof(olink));
    olnk->pos = pos;
    olnk->next = NULL;
    return olnk;
}

void free_olink(olink *olnk) {
    free(olnk);
}

void display_occ_list(olink *olst) {
    while (olst) {
        printf("%d, \n", olst->pos);
        olst = olst->next;
    }
    putchar('\n');
}

void free_occ_list(olink *olst) {
    while (olst) {
        olink *tmp = olst;
        olst = olst->next;
        free_olink(tmp);
    }
}

int equal_occ_element(int pos1, int pos2) {
    return pos1 == pos2;
}

olink *find_olist(olink* olst, int pos) {
    olink *ptr = olst;
    while (ptr != NULL && !equal_occ_element(ptr->pos, pos))
        ptr = ptr->next;
    return ptr;
}

olink *insert_last_olist(olink *olst, int pos) {
    olink *tmp = create_olink(pos);
    olink* iter_tmp = olst;

    if ( olst == NULL )
    {
    	return tmp;
    }
    else
    {
    	while ( iter_tmp->next != NULL ) iter_tmp = iter_tmp->next;
    	iter_tmp->next = tmp;
    	return olst;
    }
}

void add_occurrence(link *lnk, int pos)
{
	lnk->occurrence = insert_last_olist(lnk->occurrence, pos);
}

int get_occurence_number(olink* olink)
{
	int total_number = 0;
	while( olink != NULL )
	{
		total_number++;
		olink = olink->next;
	}

	return total_number;
}

/**
 * Create a link representing 1 occurence of the string 'word'.
 * The string is copied and must be freed when the link is freed.
 */
link *create_link(char word[], int pos) {
    link *lnk = malloc(sizeof(link));
    lnk->word = malloc(strlen(word)+1);
    strcpy(lnk->word, word);
    add_occurrence(lnk, pos);
    lnk->next = NULL;
    return lnk;
}

void free_link(link *lnk) {
    free_occ_list(lnk->occurrence);
    free(lnk);
}

void display_list(link *lst) {
    while (lst) {
        printf("%s : %d\n", lst->word, get_occurence_number(lst->occurrence));
        lst = lst->next;
    }
}

void free_list(link *lst) {
    while (lst) {
        link *tmp = lst;
        lst = lst->next;
        free_link(tmp);
    }
}

int equal(char* w1, char* w2) {
    return strcmp(w1, w2) == 0;
}

link *find_list(link* lst, char word[]) {
    link *ptr = lst;
    while (ptr != NULL && !equal(ptr->word, word))
    {
        ptr = ptr->next;
    }
    return ptr;
}

link *insert_first_list(link *lst, char word[], int pos) {
    link *tmp = create_link(word, pos);
    tmp->next = lst;
    return tmp;
}

int get_total_list_word_number(link* lst)
{
    int nbr_total = 0;
    olink* buff;

    while ( lst != NULL )
    {
        buff = lst->occurrence;
        while ( buff != NULL )
        {
            nbr_total++;
            buff = buff->next;
        }
        lst = lst->next;
    }

    return nbr_total;
}