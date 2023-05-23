#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

// #define VIRTUAL

class First {
 public:
#ifdef VIRTUAL
  First() { LOG(); }
  virtual ~First() { LOG(); }
  virtual void MyFunc()
#else
  First() { LOG(); }
  ~First() { LOG(); }
  void MyFunc()
#endif
  {
    LOG();
  }
};

class Second : public First {
 public:
#ifdef VIRTUAL
  Second() { LOG(); }
  virtual ~Second() { LOG(); }
  virtual void MyFunc()
#else
  Second() { LOG(); }
  ~Second() { LOG(); }
  void MyFunc()
#endif
  {
    LOG();
  }
};

class Third : public Second {
 public:
#ifdef VIRTUAL
  Third() { LOG(); }
  virtual ~Third() { LOG(); }
  virtual void MyFunc()
#else
  Third() { LOG(); }
  ~Third() { LOG(); }
  void MyFunc()
#endif
  {
    LOG();
  }
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
