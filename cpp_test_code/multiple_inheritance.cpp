#include <iostream>

class BaseOne {
  public:
    void SimpleFunc() { std::cout << "BaseOne" << std::endl; }
};

class BaseTwo {
  public:
    void SimpleFunc() { std::cout << "BaseTwo" << std::endl; }
};

class MultiDrived : public BaseOne, public BaseTwo {
  public:
    void SimpleFunc() {
        std::cout << "MultiDrived" << std::endl;
        BaseOne::SimpleFunc();
        BaseTwo::SimpleFunc();
    }
};

int main(void) {
    MultiDrived mdr;
    mdr.SimpleFunc();
    return 0;
}
