#include <iostream>

#define LOG() printf("%s +%d\t%s\n", __FILE__, __LINE__, __PRETTY_FUNCTION__)

using namespace std;

class DataCopy {
  // Sample 01: Private Data Member
 private:
  int* x;

 public:
  // Sample 02: Constructor with single parameter
  DataCopy() {
    LOG();
    x = new int;
  }

  // Sample 08: Introduce Copy Constructor and perform Deep Copy
  // Add 'explicit' to avoid assign operation
  // Copy Constructor
  explicit DataCopy(const DataCopy& obj) {
    // DataCopy(const DataCopy& obj) {
    LOG();
    x = new int;
    *x = obj.getX();
  }

  // Copy Assignment Operator
  DataCopy& operator=(const DataCopy& obj) {
    LOG();
    x = new int;
    *x = obj.getX();
    return *this;
  }

  // Sample 03: Get and Set Functions
  int getX() const { return *x; }
  void setX(int m) { *x = m; }

  // Sample 04: Print Function
  void PrintX() { cout << "Int X=" << *x << endl; }

  // Sample 05: DeAllocate the heap
  ~DataCopy() {
    LOG();
    delete x;
  }
};

int main() {
  DataCopy obj1;
  obj1.setX(10);
  obj1.PrintX();

  // Copy Constructor
  DataCopy obj2(obj1);
  obj2.PrintX();

  // Copy Assignment Operator
  DataCopy obj3;
  obj3 = obj1;
  obj3.PrintX();
}
