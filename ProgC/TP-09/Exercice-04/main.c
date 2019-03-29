#include <stdio.h>
#include <stdlib.h>

void print_info_zone( void* adr )
{
	size_t* info_adr = (size_t*)adr;

	printf("Zone a l'adresse : %p \n", adr);

	printf("%lu\n", info_adr[-1] );

	printf("%x\n", info_adr[-2] );

}


/* malloc est genereux --" */
void print_between_zone( void* adr1, void* adr2 )
{
	unsigned char i = 0;
	char* adr1_char_view = adr1;
	for ( i = 0 ; i < 128 ; i++ )
	{
		printf("%x ", *(adr1_char_view+i));
	}

	putchar('\n');
	printf("case 0 = %lu\n", *(adr1_char_view+136));
	printf("case 1 = %lu\n", *(adr1_char_view+144));
}

int main(void)
{
	int i;
	char* escroc = (char*)malloc(128);
	int* escroc2 = malloc(1);
	for ( i = 0 ; i < 136 ; i++ )
	{
		escroc[i] = i;
	}
	printf("val(escroc) = %x\n", (unsigned int)*escroc);
	print_between_zone(escroc,escroc2);

	print_info_zone(escroc);
	print_info_zone(escroc2);
	free(escroc);
	free(escroc);
	free(escroc2);
	return 0;
}