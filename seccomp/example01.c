/*
 * example01.c
 * Copyright (c) 2016 Xianglan Piao <xianglan0502@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * File: example01.c
 * Author: Xianglan Piao <xianglan0502@gmail.com>
 * Date: 2016.10.07
 */
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
