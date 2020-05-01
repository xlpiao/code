#include <iostream>

using namespace std;

// #define STATIC
#ifdef STATIC
class SoSimple {
 private:
  int num1 = 0;
  static int num2;

 public:
  SoSimple(int n) : num1(n) {}
  static void Adder(int n) {
    // num1 += n;    // 컴파일 에러!!
    num2 += n;
    // std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;
  }
};
int SoSimple::num2 = 0;  // static variable must initialized outside class

int main(void) {
  SoSimple s(10);
  s.Adder(2);
  s.Adder(2);
  s.Adder(2);

  return 0;
}
#endif

// #define CONST
#ifdef CONST
class SoSimple {
 private:
  int num1;
  const int num2;

 public:
  SoSimple(int n) : num1(n), num2(n) {}
  void AddNum(int n) {
    // SoSimple& AddNum(int n) {
    // SoSimple& AddNum(int n) const {
    // num1 += n;
    num2 += n;  // 컴파일 에러!!
    std::cout << num1 << std::endl;
    std::cout << num2 << std::endl;
    // return *this;
  }
};

int main(void) {
  SoSimple s(10);
  // const SoSimple s(10);
  s.AddNum(2);
  s.AddNum(2);
  s.AddNum(2);

  return 0;
}
#endif

// #define CONST_STATIC
#ifdef CONST_STATIC
class CountryArea {
 public:
  const static int RUSSIA = 1;
  const static int CANADA = 2;
  const static int CHINA = 3;
};

int main(void) {
  std::cout << "Russion Area: " << CountryArea::RUSSIA << std::endl;
  std::cout << "Canada Area: " << CountryArea::CANADA << std::endl;
  std::cout << "China Area: " << CountryArea::CHINA << std::endl;
  std::cout << "Russion Area: " << CountryArea::RUSSIA << std::endl;
  std::cout << "Canada Area: " << CountryArea::CANADA << std::endl;
  std::cout << "China Area: " << CountryArea::CHINA << std::endl;

  return 0;
}
#endif

// #define MUTABLE
#ifdef MUTABLE
/* mutable는 const 키워드를 의미 없게 만들 수 있다 */
class SoSimple {
 private:
  int num1;
  mutable int num2;

 public:
  SoSimple(int n1, int n2) : num1(n1), num2(n2) {}
  void CopyToNum2() const {
    std::cout << num1 << ", " << num2 << std::endl;
    num2 = num1;
    std::cout << num1 << ", " << num2 << std::endl;
  }
};

int main(void) {
  SoSimple s(1, 2);
  s.CopyToNum2();
}
#endif

// #define VOLATILE
#ifdef VOLATILE
/* 00. original code */
int cond = 20;
while (cond == 20) {
  /* condition is true */
}

/* 01. after compile */
// int cond = 20; // Unless used, this will be removed too.
while (true) {
  /* condition is true */
}

/* 02. with volatile qualifier, 03 will not work on it */
volatile int cond = 20;
while (cond == 20) {
  // condition is true
}
#endif

#define ATOMIC
#ifdef ATOMIC
/* clang++ type_qualifierr.cpp -lpthread -std=c++11 */
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class SoSimple {
 public:
  std::atomic<int> count;
  // int count;    // Not safe at all to use in multithreaded environment.
  SoSimple() : count(0) {}

  void increment() {
    ++count;
    std::cout << "tid: " << std::this_thread::get_id() << ", value: " << count
              << std::endl;
  }

  void decrement() {
    --count;
    std::cout << "tid: " << std::this_thread::get_id() << ", value: " << count
              << std::endl;
  }

  int get() { return count.load(); }  // std::atomic<int> count
  // int get() { return count; }          // int count
};

int main(void) {
  SoSimple s;

  std::vector<std::thread> threads;
  for (int i = 0; i < 2; ++i) {
    threads.push_back(std::thread([&s]() {
      for (int j = 0; j < 5; ++j) {
        s.increment();
      }
    }));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::cout << s.get() << std::endl;
}
#endif
