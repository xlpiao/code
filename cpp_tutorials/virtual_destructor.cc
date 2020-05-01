#include <iostream>

// #define VIRTUAL

class First {
 public:
#ifdef VIRTUAL
  First() { std::cout << "First Constructor" << std::endl; }
  virtual ~First() { std::cout << "First Destructor" << std::endl; }
  virtual void MyFunc()
#else
  First() { std::cout << "First Constructor" << std::endl; }
  ~First() { std::cout << "First Destructor" << std::endl; }
  void MyFunc()
#endif
  {
    std::cout << "FirstFunc" << std::endl;
  }
};

class Second : public First {
 public:
#ifdef VIRTUAL
  Second() { std::cout << "Second Constructor" << std::endl; }
  virtual ~Second() { std::cout << "Second Destructor" << std::endl; }
  // 오버라이딩 된 함수가 virtual이면 오버라이딩 한 함수도 자동으로 virtual
  virtual void MyFunc()
#else
  Second() { std::cout << "Second Constructor" << std::endl; }
  ~Second() { std::cout << "Second Destructor" << std::endl; }
  void MyFunc()
#endif
  {
    std::cout << "SecondFunc" << std::endl;
  }
};

class Third : public Second {
 public:
#ifdef VIRTUAL
  Third() { std::cout << "Third Constructor" << std::endl; }
  virtual ~Third() { std::cout << "Third Destructor" << std::endl; }
  virtual void MyFunc()
#else
  Third() { std::cout << "Third Constructor" << std::endl; }
  ~Third() { std::cout << "Third Destructor" << std::endl; }
  void MyFunc()
#endif
  {
    std::cout << "ThirdFunc" << std::endl;
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
