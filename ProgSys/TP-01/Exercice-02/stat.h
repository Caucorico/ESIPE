#ifndef _STAT_
#define _STAT_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#define _BSD_SOURCE 200112L

int get_struct_stat(const char* path, struct stat* file_stat);

int display_struct_stat(const struct stat* file_stat, const char* path);

#endif
