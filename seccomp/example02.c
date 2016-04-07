#include <stdio.h>      // printf 
#include <unistd.h>     // dup2: just for test
#include <seccomp.h>    // libseccomp


int main() {
    printf("step 1: unrestricted\n");

    scmp_filter_ctx ctx;  // Init the filter
    ctx = seccomp_init(SCMP_ACT_KILL);    // default action: kill

    // setup basic whitelist
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(rt_sigreturn), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);

    // setup our rule
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(dup2), 2, 
                                          SCMP_A0(SCMP_CMP_EQ, 1),
                                          SCMP_A1(SCMP_CMP_EQ, 2));

    // build and load the filter
    seccomp_load(ctx);
    printf("step 2: only 'write' and dup2(1, 2) syscalls\n");

    // Redirect stderr to stdout
    dup2(1, 2);
    printf("step 3: stderr redirected to stdout\n");

    // Duplicate stderr to arbitrary fd
    dup2(2, 42);
    printf("step 4: !! YOU SHOULD NOT SEE ME !!\n");

    seccomp_release(ctx);

    return 0;
}
