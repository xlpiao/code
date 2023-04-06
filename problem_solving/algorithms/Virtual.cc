#include <iostream>
using namespace std;

class Base {
 public:
  Base() { cout << "Base Constructor" << endl; }
  ~Base() { cout << "Base Destructor" << endl; }
  virtual void print1() { cout << "Base print1" << endl; }
  void print2() { cout << "Base print2" << endl; }
};

class Derived : public Base {
 public:
  Derived() { cout << "Derived Constructor" << endl; }
  ~Derived() { cout << "Derived Destructor" << endl; }
  void print1() { cout << "Derived print1" << endl; }
  void print2() { cout << "Derived print2" << endl; }
};

int main() {
  Derived derived;
  Base* base1 = &derived;

  base1->print1();
  base1->print2();

  return 0;
}
