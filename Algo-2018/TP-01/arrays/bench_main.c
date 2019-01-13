#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include "benchmark.h"

void usage() {
	printf("Usage: bench [options]\n");
	printf("  -a               Run all tests.\n");
	printf("  -h               Print this message and exit.\n");
	printf("  -r               Run only reference functions.\n");
	printf("  -s               Run only tests on sorted arrays.\n");
	printf("  -t [N]           Run only test number N [1..%d].\n", N_TESTS);
	printf("  -u               Run only tests on unsorted arrays.\n");
}

int main(int argc, char* argv[]) {

	int rflag = 0;
	int uflag = 1;
	int sflag = 1;
	int c;
	int nb = 0;
	int first = 1, last = N_TESTS;

	if (argc == 1) {
		usage();
		return 0;
	}

	while ((c = getopt(argc, argv, "ahrst:u")) != -1)
		switch (c) {
			case 'a':
			    break;
			case 'h':
			    usage();
			    return 0;
			case 'r':
				rflag = 1;
				break;
			case 's':
				uflag = 0;
				break;
			case 't':
				nb = atoi(optarg);
				if (nb < 1 || nb > N_TESTS) {
					usage();
					return 1;
				}
				break;
			case 'u':
				sflag = 0;
				break;
			case '?':
				usage();
		        return 1;
		    default:
		    	abort();
		}

	if (nb != 0) {
		first = nb;
		last = nb;
	}

	int res1, res2;
	int passed = 0;
	int mode = 0;
	if (rflag) {
		mode = MODE_REFONLY;
	}
	int i;
	for (i = first; i <= last; i++) {
		res1 = 1;
		res2 = 1;
		if (uflag) {
			res1 = benchmark(i, mode);
		}
		if (sflag) {
			res2 = benchmark(i, mode | MODE_SORTED);
		}
		if (res1 && res2) {
			passed++;
		}
	}
	if (!rflag)
		printf("%d of %d tests passed\n", passed, last-first+1);

	return 0;
}