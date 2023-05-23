#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

using namespace std;

// #define VIRTUAL

class First {
 public:
#ifdef VIRTUAL
  virtual void MyFunc()
#else
  void MyFunc()
#endif
  {
    LOG();
  }
};

class Second : public First {
 public:
#ifdef VIRTUAL
  // 오버라이딩 된 함수가 virtual이면 오버라이딩 한 함수도 자동으로 virtual
  virtual void MyFunc()
#else
  void MyFunc()
#endif
  {
    LOG();
  }
};

class Third : public Second {
 public:
#ifdef VIRTUAL
  virtual void MyFunc()
#else
  void MyFunc()
#endif
  {
    LOG();
  }
};

int main(void) {
  Third* third = new Third();
  Second* second = third;
  First* first = second;

  third->MyFunc();
  second->MyFunc();
  first->MyFunc();

  delete third;

  return 0;
}
