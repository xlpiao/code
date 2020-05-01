#include <iostream>
using namespace std;

#define VIRTUAL

class First {
 public:
#ifdef VIRTUAL
  virtual void MyFunc()
#else
  void MyFunc()
#endif
  {
    cout << "FirstFunc" << endl;
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
    cout << "SecondFunc" << endl;
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
    cout << "ThirdFunc" << endl;
  }
};

int main(void) {
  Third* tptr = new Third();
  Second* sptr = tptr;
  First* fptr = sptr;

  tptr->MyFunc();
  sptr->MyFunc();
  fptr->MyFunc();

  delete tptr;

  return 0;
}
