#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

#define VIRTUAL

class First {
 public:
#ifdef VIRTUAL
  First() { LOG(); }
  virtual ~First() { LOG(); }
  virtual void MyFunc() { LOG(); }
#else
  First() { LOG(); }
  ~First() { LOG(); }
  void MyFunc() { LOG(); }
#endif
};

class Second : public First {
 public:
#ifdef VIRTUAL
  Second() { LOG(); }
  ~Second() { LOG(); }
  void MyFunc() override { LOG(); }
#else
  Second() { LOG(); }
  ~Second() { LOG(); }
  void MyFunc() { LOG(); }
#endif
};

class Third : public Second {
 public:
#ifdef VIRTUAL
  Third() { LOG(); }
  ~Third() { LOG(); }
  void MyFunc() override { LOG(); }
#else
  Third() { LOG(); }
  ~Third() { LOG(); }
  void MyFunc() { LOG(); }
#endif
};

int main(void) {
  std::cout << "3" << std::endl;
  Third* third = new Third();
  third->MyFunc();

  std::cout << "2" << std::endl;
  Second* second = third;
  second->MyFunc();

  std::cout << "1" << std::endl;
  First* first = second;
  first->MyFunc();

  delete third;

  return 0;
}
