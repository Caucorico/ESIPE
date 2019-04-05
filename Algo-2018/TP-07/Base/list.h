#ifndef __LIST_H__
#define __LIST_H__

typedef struct _olink {
	int pos;
	struct _olink *next;
} olink;

typedef struct _link {
    char *word;
    olink* occurrence;
    struct _link *next;
} link;

void free_list(link *lst);

link *find_list(link *lst, char word[]);
link *insert_first_list(link *lst, char word[], int pos);

void display_list(link *lst);

void add_occurrence(link *lnk, int pos);

#endif
