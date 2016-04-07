#include <stdio.h>          // printf
#include <sys/prctl.h>      // prctl
#include <linux/seccomp.h>  // seccomp's constants
#include <unistd.h>         // dup2: just for test

int main() {
    printf("step 1: unrestricted\n");

    prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);   // Enable filtering
    printf("step 2: only 'read', 'write', '_exit' and 'sigreturn' syscalls\n");

    dup2(1, 2);   // Redirect stderr to stdout
    printf("step 3: !! YOU SHOULD NOT SEE ME !!\n");

    return 0; 
}
