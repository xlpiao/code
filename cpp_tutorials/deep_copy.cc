#include <iostream>
using namespace std;

class DataCopy {
  // Sample 01: Private Data Member
 private:
  int* x;

 public:
  // Sample 02: Constructor with single parameter
  DataCopy(int m) {
    std::cout << "DataCopy(int m)" << std::endl;
    x = new int;
    *x = m;
  }

  // Sample 08: Introduce Copy Constructor and perform Deep Copy
  // Add 'explicit' to avoid assign operation
  explicit DataCopy(const DataCopy& obj) {
    // DataCopy(const DataCopy& obj) {
    std::cout << "DataCopy(const DataCopy& obj)" << std::endl;
    x = new int;
    *x = obj.GetX();
  }

  DataCopy& operator=(const DataCopy& obj) {
    std::cout << "DataCopy ob2 = ob1" << std::endl;
    x = new int;
    *x = obj.GetX();
    return *this;
  }

  // Sample 03: Get and Set Functions
  int GetX() const { return *x; }
  void SetX(int m) { *x = m; }

  // Sample 04: Print Function
  void PrintX() { cout << "Int X=" << *x << endl; }

  // Sample 05: DeAllocate the heap
  ~DataCopy() {
    std::cout << "~DataCopy()" << std::endl;
    delete x;
  }
};

int main() {
  DataCopy ob1(10);
  ob1.PrintX();

  /* remove explicit in copy constructor */
  /* Can user operator to enable this */
  DataCopy ob2 = ob1;
  ob2.PrintX();

  DataCopy ob3(ob1);
  ob3.PrintX();
}
