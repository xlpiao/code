#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

using namespace std;

class Base {
 public:
  Base() { LOG(); }
  ~Base() { LOG(); }
  virtual void print1() { LOG(); }
  void print2() { LOG(); }
};

class Derived : public Base {
 public:
  Derived() { LOG(); }
  ~Derived() { LOG(); }
  void print1() { LOG(); }
  void print2() { LOG(); }
};

int main() {
  Derived derived;
  Base* base1 = &derived;

  base1->print1();
  base1->print2();

  return 0;
}
