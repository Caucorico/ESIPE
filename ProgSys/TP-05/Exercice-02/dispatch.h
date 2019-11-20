#ifndef _DISPATCH_
#define _DISPATCH_

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

/* This function create a proc which can run a function. */
int startProc(int (*function)(int, int), int* read_fd, int* write_fd);

#endif
