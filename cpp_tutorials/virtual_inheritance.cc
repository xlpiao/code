#include <iostream>

class Base {
 public:
  Base() { std::cout << "Base" << std::endl; };
  virtual void func() { std::cout << "Base::func" << std::endl; };
};

class DerivedA : virtual public Base {
 public:
  DerivedA() { std::cout << "DerivedA" << std::endl; };
  void func() { std::cout << "DerivedA::func" << std::endl; };
};

// class DerivedB : virtual public Base, public DerivedA {
class DerivedB : virtual public Base, virtual public DerivedA {
 public:
  DerivedB() { std::cout << "DerivedB" << std::endl; };
  void func() { std::cout << "DerivedB::func" << std::endl; };
};

int main(int argc, char *argv[]) {
  Base *b1;
  Base *b2;
  DerivedA *d1 = new DerivedA();
  DerivedB *d2 = new DerivedB();

  b1 = d1;
  b2 = d2;

  b1->func();
  b2->func();
}
