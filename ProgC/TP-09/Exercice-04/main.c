#include <stdio.h>
#include <stdlib.h>

void print_info_zone( void* adr )
{
	size_t* info_adr = (size_t*)adr;

	printf("Zone a l'adresse : %p \n", adr);

	printf("%lu\n", info_adr[-1] );

	printf("%lu\n", info_adr[-2] );
}

int main(void)
{
	int* escroc = malloc(1);
	int* escroc2 = malloc(1);
	print_info_zone(escroc);
	print_info_zone(escroc2);
	free(escroc);
	free(escroc2);
	return 0;
}