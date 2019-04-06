#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "list.h"

#define MAX_WORD_LENGTH 80

link* insert_word_in_list( link* lst, char* word, int pos )
{
	link* element;
	element = find_list(lst, word);

	if ( element != NULL )
	{
		add_occurrence(element, pos);
	}
	else
	{
		lst = insert_first_list(lst, word, pos);
	}

	return lst;
}

link *read_text(FILE *infile) {
	link* lst = NULL;
    char *word = (char *)malloc(MAX_WORD_LENGTH*sizeof(char));
    int word_nbr = 0;

    while (fscanf(infile, "%s ", word) != -1)
    {
        lst = insert_word_in_list(lst, word, word_nbr);
        word_nbr++;
    }
    free(word);
    return lst;
}

int get_total_word_number(link* lst)
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

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Usage: concordance <in_file>\n");
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (fin == NULL) {
        fprintf(stderr, "Error opening file for reading: %s\n", argv[1]);
        return 1;
    }

    link *lst = read_text(fin);
    fclose(fin);

    display_list(lst);
    
    int words = 0;
    link *ptr;
    for (ptr = lst; ptr != NULL; ptr = ptr->next) {
        words++;
    }

    printf("total number of word : %d\n", get_total_word_number(lst));

    printf("total number of distinct words = %d\n", words);

    free_list(lst);

    return 0;
}