#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

class Base {
 public:
  Base() { LOG(); };
  ~Base() { LOG(); };
  virtual void func() { LOG(); };
};

class Derived1 : virtual public Base {
 public:
  Derived1() { LOG(); };
  ~Derived1() { LOG(); };
  void func() { LOG(); };
};

// class Derived2 : virtual public Base, public Derived1 {
class Derived2 : virtual public Base, virtual public Derived1 {
 public:
  Derived2() { LOG(); };
  ~Derived2() { LOG(); };
  void func() { LOG(); };
};

int main(int argc, char *argv[]) {
  Derived1 d1;
  Derived2 d2;

  std::cout << "1" << std::endl;
  Base &b1 = d1;
  b1.func();

  std::cout << "2" << std::endl;
  Base &b2 = d2;
  b2.func();
}
