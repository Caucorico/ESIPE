#ifndef _DIRSTAT_
#define _DIRSTAT_

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include "stat.h"

/* extern DFN; Display File Name */
/* extern DCF; Display File Content */

int get_dir_struct_stat(const char* path, struct dirent*** dir_stat);

int close_dir_struct_stat(struct dirent*** dir_stat);

int display_dir_struct_stat(struct dirent* dir_stat[], int nbr_element, unsigned char flags);

#endif
