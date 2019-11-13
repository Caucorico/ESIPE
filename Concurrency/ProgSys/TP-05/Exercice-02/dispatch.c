#include "dispatch.h"

void runProc(int(*function)(int, int), int read_fd, int write_fd)
{
	char buff[4];
	int err;
	int a, b;

	while(1)
	{
		fscanf(read_fd, "%d %d", &a, &b);
		res = (*function)(a, b);
		fprintf(write_fd, "%d", res);
	}
}

int startProc(int(*function)(int, int), int* read_fd, int* write_fd)
{
	int err, proc;
	int pipe_f_s[2];
	int pipe_s_f[2];

	err = pipe(pipe_f_s);
	if ( err < 0 )
	{
		perror("pipe error ");
		return -1;
	}

	err = pipe(pipe_s_f);
	if ( err < 0 )
	{
		perror("pipe error 2 ");
		close(pipe_f_s[0]);
		close(pipe_f_s[1]);
		return -2;
	}

	proc = fork();

	switch(proc)
	{
		case -1:
			perror("fork error ");
			return -3;
		case 0:
			close(pipe_f_s[1]);
			close(pipe_s_f[0]);
			runProc(function, pipe_f_s[0], pipe_s_f[1]);
			break;

		default:
			*read_fd = pipe_s_f[0];
			close(pipe_s_f[1]);
			*write_fd = pipe_f_s[1];
			close(pipe_f_s[0]);
			return proc;
	}

	return 0;
}
