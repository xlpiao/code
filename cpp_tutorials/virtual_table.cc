#include <iostream>

class AAA {
 private:
  int num1;

 public:
  virtual void Func1() { std::cout << "AAA::Func1" << std::endl; }
  virtual void Func2() { std::cout << "AAA::Func2" << std::endl; }
};

class BBB : public AAA {
 private:
  int num2;

 public:
  virtual void Func1() { std::cout << "BBB::Func1" << std::endl; }
  void Func3() { std::cout << "BBB::Func3" << std::endl; }
};

int main(void) {
  AAA* aptr = new AAA();
  aptr->Func1();

  BBB* bptr = new BBB();
  bptr->Func1();
  return 0;
}
