#include <iostream>

class First {
  public:
    void FirstFunc() { std::cout << "FirstFunc" << std::endl; }
};

class Second : public First {
  public:
    void SecondFunc() { std::cout << "SecondFunc" << std::endl; }
};

class Third : public Second {
  public:
    void ThirdFunc() { std::cout << "ThirdFunc" << std::endl; }
};

int main(void) {
    Third* third_ptr = new Third();
    /* Third형 포인터 변수 third_ptr이 가리키는 객체는 무조건 Second형 포인터
     * 변수 second_ptr도 가리킬수 있음 */
    Second* second_ptr = third_ptr;
    /* Second형 포인터 변수 second_ptr이 가리키는 객체는 무조건 First형 포인터
     * 변수 first_ptr도 가리킬 수 있음 */
    First* first_ptr = second_ptr;

    third_ptr->FirstFunc();
    third_ptr->SecondFunc();
    third_ptr->ThirdFunc();

    second_ptr->FirstFunc();
    second_ptr->SecondFunc();
    // second_ptr->ThirdFunc();    // error: no member named 'ThirdFunc' in 'Second'

    first_ptr->FirstFunc();
    // first_ptr->SecondFunc();    // error: no member named 'SecondFunc' in 'First'
    // first_ptr->ThirdFunc();     // error: no member named 'ThirdFunc' in 'First'

    delete third_ptr;
    return 0;
}
