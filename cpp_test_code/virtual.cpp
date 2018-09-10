#include <iostream>
using namespace std;

class First {
  public:
    void FirstFunc() {
        cout << "FirstFunc()" << endl;
    }
    virtual void SimpleFunc() {
        cout << "First's SimpleFunc()" << endl;
    }
};

class Second : public First {
  public:
    void SecondFunc() {
        cout << "SecondFunc()" << endl;
    }
    virtual void SimpleFunc() {
        cout << "Second's SimpleFunc()" << endl;
    }
};

class Third : public Second {
  public:
    void ThirdFunc() {
        cout << "ThirdFunc()" << endl;
    }
    virtual void SimpleFunc() {
        cout << "Third's SimpleFunc()" << endl;
    }
};

int main(void) {
    Third obj;
    obj.FirstFunc();
    obj.SecondFunc();
    obj.ThirdFunc();
    obj.SimpleFunc();

    Second& sref = obj;    // obj는 Second를 상속하는 Third 객체이므로, Second형 참조자로 참조 가능
    sref.FirstFunc();
    sref.SecondFunc();
    // 컴파일러는 참조자의 자료형을 가지고 함수의 호출 가능성을 판단
    // First 클래스에 정의된 FirstFunc 함수와 Second 클래스에 정의된 SecondFunc 함수는 호출 가능

    sref.SimpleFunc();
    // SimpeFunc 함수는 가상함수 -> Third 클래스에 정의된 SimpleFunc 함수가 호출

    First& fref = obj;
    // obj는 First를 간접 상속하는 Third 객체이므로, First형 참조자로 참조 가능

    fref.FirstFunc();
    fref.SimpleFunc();
    // First 클래스에 정의된 함수만 호출 가능. SimpleFunc는 가상함수 -> Third 클래스에 정의된 SimpleFunc 함수 호출
    return 0;
}
