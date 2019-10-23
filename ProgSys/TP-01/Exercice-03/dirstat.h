#ifndef _DIRSTAT_
#define _DIRSTAT_

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include "stat.h"

const DFN = 0x1; /* Display File Name */
const DCF = 0x2; /* Display File Content */

int get_dir_struct_stat(const char* path, struct dirent*** dir_stat);

int close_dir_struct_stat(const dirent*** dir_stat);

int display_dir_struct_stat(const struct dirent* dir_stat[], int nbr_element, unsigned char flags);

#endif
