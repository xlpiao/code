#include <stddef.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <sys/syscall.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    pid_t pid;

    struct sock_filter filter[] = {
        /* Load architecture */
        BPF_STMT(BPF_LD+BPF_W+BPF_ABS, (offsetof(struct seccomp_data, arch))),

        /* Kill process if the architecture is not what we expect */
        /* AUDIT_ARCH_X86_64 or AUDIT_ARCH_I386 */
        BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_KILL),

        /* Load system call number */
        BPF_STMT(BPF_LD+BPF_W+BPF_ABS, (offsetof(struct seccomp_data, nr))),

        /* Allow system calls other than open() */
        BPF_JUMP(BPF_JMP+BPF_JEQ+BPF_K, __NR_open, 1, 0),
        BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_ALLOW),

        /* Kill process on open() */
        BPF_STMT(BPF_RET+BPF_K, SECCOMP_RET_KILL)
    };

    struct sock_fprog prog = {
        .len = (unsigned short) (sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };

    printf("\n*** Appling BPF filter ***\n\n");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
        perror("prctl");
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog))
        perror("seccomp");

    pid=fork();
    if(pid==0)
    {
        printf("I am the child\n");
        printf("The PID of child is %d\n",getpid());
        printf("The PID of parent of child is %d\n\n",getppid());
    }
    else
    {
        printf("I am the parent\n");
        printf("The PID of parent is %d\n",getpid());
        printf("The PID of parent of parent is %d\n\n",getppid());        
    }

    exit(EXIT_SUCCESS);
}
