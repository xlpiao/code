#include <iostream>
using namespace std;

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

class First {
 public:
  First() { LOG(); }
  ~First() { LOG(); }
  void Func() { LOG(); }
  virtual void SimpleFunc() { LOG(); }
};

class Second : public First {
 public:
  Second() { LOG(); }
  ~Second() { LOG(); }
  void Func() { LOG(); }
  virtual void SimpleFunc() { LOG(); }
};

class Third : public Second {
 public:
  Third() { LOG(); }
  ~Third() { LOG(); }
  void Func() { LOG(); }
  void SimpleFunc() override { LOG(); }
};

int main(void) {
  cout << "1" << endl;
  Third obj;
  obj.Func();
  obj.SimpleFunc();

  cout << "2" << endl;
  Second& s = obj;
  s.Func();
  s.SimpleFunc();

  cout << "3" << endl;
  First& f = obj;
  f.Func();
  f.SimpleFunc();
  return 0;
}
