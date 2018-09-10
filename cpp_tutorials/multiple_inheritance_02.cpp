#include <iostream>
using namespace std;

class Base {
  public:
    Base() { cout << "Base Constructor" << endl; }
    void SimpleFunc() { cout << "BaseOne" << endl; }
};

class MiddleDrivedOne : public Base {
  public:
    MiddleDrivedOne() : Base() {
        cout << "MiddleDrivedOne Constructor" << endl;
    }
    void MiddleFuncOne() {
        SimpleFunc();
        cout << "MiddleDrivedOne" << endl;
    }
    void SimpleFunc() { cout << "who am I?? one??" << endl; }
};

class MiddleDrivedTwo : public Base {
  public:
    MiddleDrivedTwo() : Base() {
        cout << "MiddleDrivedTwo Constructor" << endl;
    }
    void MiddleFuncTwo() {
        SimpleFunc();
        cout << "MiddleDrivedTwo" << endl;
    }
    void SimpleFunc() { cout << "who am I?? two??" << endl; }
};

class LastDerived : public MiddleDrivedOne, public MiddleDrivedTwo {
  public:
    LastDerived() : MiddleDrivedOne(), MiddleDrivedTwo() {
        cout << "LastDerived Constructor" << endl;
    }
    void ComplexFunc() {
        MiddleFuncOne();
        MiddleFuncTwo();
        SimpleFunc();
    }
    void SimpleFunc() { cout << "who am I?? LastDerived??" << endl; }
};

int main(void) {
    cout << "객체생성 전 ..... " << endl;
    LastDerived ldr;
    cout << "객체생성 후 ..... " << endl;
    ldr.ComplexFunc();
    return 0;
}
