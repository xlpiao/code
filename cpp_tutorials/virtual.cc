#include <iostream>
using namespace std;

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

class Parent {
 public:
  Parent() { LOG(); }
  virtual ~Parent() { LOG(); }
  void Func() { LOG(); }
  virtual void SimpleFunc() { LOG(); }
};

class Child1 : public Parent {
 public:
  Child1() { LOG(); }
  ~Child1() { LOG(); }
  void Func() { LOG(); }
  virtual void SimpleFunc() { LOG(); }
};

class Child2 : public Child1 {
 public:
  Child2() { LOG(); }
  ~Child2() { LOG(); }
  void Func() { LOG(); }
  void SimpleFunc() { LOG(); }
};

int main(void) {
  cout << "1" << endl;
  Child1* c1 = new Child1();
  c1->Func();
  c1->SimpleFunc();
  // delete c1;

  cout << "2" << endl;
  Child2* c2 = new Child2();
  c2->Func();
  c2->SimpleFunc();
  // delete c2;

  cout << "3" << endl;
  Parent* f = c1;
  f->Func();
  f->SimpleFunc();
  delete f;

  cout << "4" << endl;
  Parent* test = new Child2();
  test->Func();
  test->SimpleFunc();
  delete test;

  cout << "delete" << endl;
  // delete c1;
  delete c2;

  return 0;
}
