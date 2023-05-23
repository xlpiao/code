#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)
using namespace std;

class AAA {
 public:
  AAA() { LOG(); }
  ~AAA() { LOG(); }
  virtual void Func1() { LOG(); }
  void Func2() { LOG(); }
};

class BBB : public AAA {
 public:
  BBB() { LOG(); }
  ~BBB() { LOG(); }
  virtual void Func1() { LOG(); }
  void Func2() { LOG(); }
  void Func3() { LOG(); }
};

int main(void) {
  cout << "1" << endl;
  AAA a;
  a.Func1();
  a.Func2();

  cout << "2" << endl;
  BBB b;
  b.Func1();
  b.Func2();

  cout << "3" << endl;
  AAA* obj = &b;
  obj->Func1();
  obj->Func2();

  cout << "4" << endl;
  AAA& ref = b;
  ref.Func1();
  ref.Func2();

  return 0;
}
