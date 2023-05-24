#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

using namespace std;

class Base {
 public:
  Base() { LOG(); }
  virtual ~Base() { LOG(); }
  virtual void show() = 0;
};

class Derived : public Base {
 public:
  Derived() { LOG(); }
  ~Derived() { LOG(); }
  void show() { LOG(); }
};

int main(void) {
  Base *b = new Derived();
  b->show();

  delete b;

  return 0;
}
