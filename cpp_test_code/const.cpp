#include <iostream>

int main(void)
{
    int val0 = 0;
    int* ptr0 = &val0;
    std::cout << val0 << std::endl;
    std::cout << *ptr0 << std::endl;

    const int val1 = 10;
    const int* ptr1 = &val1;
    std::cout << *ptr1 << std::endl;

    // *ptr1 = 7; // error: read-only variable is not assignable
    // std::cout << *ptr1 << std::endl;

    // val1 = 11; // error: cannot assign to variable 'val1' with const-qualified type 'const int'
    // std::cout << *ptr1 << std::endl;

    ptr1 = &val0;
    std::cout << *ptr1 << std::endl;

    // const int val2 = 20;
    // int* const ptr2 = &val2; // error: cannot initialize a variable of type 'int *const' with an rvalue of type 'const int *'

    int val3 = 30;
    int* const ptr3 = &val3;
    std::cout << *ptr3 << std::endl;

    int val4 = 40;
    // ptr3 = &val4; // error: cannot assign to variable 'ptr3' with const-qualified type 'int *const'
    // std::cout << *ptr3 << std::endl;
    *ptr3 = 33;
    std::cout << *ptr3 << std::endl;
}
