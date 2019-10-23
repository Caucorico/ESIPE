#ifndef _STAT_
#define _STAT_

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

int get_struct_stat(const char* path, struct stat* file_stat);

int display_struct_stat(const struct stat* file_stat);

#endif
