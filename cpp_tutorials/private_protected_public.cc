#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

class Base {
 private:
  int num1 = 1;

 private:
  void privateFunc() {
    LOG();
    std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }

 protected:
  int num2 = 2;

 protected:
  void protectedFunc() {
    LOG();
    std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }

 public:
  int num3 = 3;

 public:
  void publicFunc() {
    LOG();
    std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }
};

#define PRIVATE
#ifdef PRIVATE
class Derived1 : private Base {
 public:
  void showBaseData() {
    LOG();
    // std::cout << num1 << std::endl;  // error: 'num1' is a private member of
    // 'Base'
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }
};
#endif

#define PROTECT
#ifdef PROTECT
class Derived2 : protected Base {
 public:
  void showBaseData() {
    LOG();
    // std::cout << num1 << std::endl;  // error: 'num1' is a private member
    // of 'Base'
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }
};
#endif

#define PUBLIC
#ifdef PUBLIC
class Derived3 : public Base {
 public:
  void showBaseData() {
    LOG();
    // std::cout << num1 << std::endl; // error: 'num1' is a private member
    // of 'Base'
    std::cout << num2 << std::endl;
    std::cout << num3 << std::endl;
  }
};
#endif

int main(void) {
#ifdef PRIVATE
  Derived1 d1;
  d1.showBaseData();
  // std::cout << d1.num1 << std::endl;
  // std::cout << d1.num2 << std::endl;
  // std::cout << d1.num3 << std::endl;
#endif

#ifdef PROTECT
  Derived2 d2;
  d2.showBaseData();
  // std::cout << d2.num1 << std::endl;
  // std::cout << d2.num2 << std::endl;
  // std::cout << d2.num3 << std::endl;
#endif

#ifdef PUBLIC
  Derived3 d3;
  d3.showBaseData();
  // std::cout << d3.num1 << std::endl;
  // std::cout << d3.num2 << std::endl;
  // std::cout << d3.num3 << std::endl;
#endif

  return 0;
}
