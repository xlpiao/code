#include <iostream>

class BaseOne {
 public:
  BaseOne() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  ~BaseOne() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  void SimpleFunc() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
};

class BaseTwo {
 public:
  BaseTwo() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  ~BaseTwo() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  void SimpleFunc() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
};

class MultiDrived : public BaseOne, public BaseTwo {
 public:
  MultiDrived() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  ~MultiDrived() { std::cout << __PRETTY_FUNCTION__ << std::endl; }
  void SimpleFunc() {
    std::cout << "MultiDrived" << std::endl;
    BaseOne::SimpleFunc();
    BaseTwo::SimpleFunc();
  }
};

int main(void) {
  MultiDrived *obj = new MultiDrived();
  obj->SimpleFunc();
  delete obj;
  return 0;
}
