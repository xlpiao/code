#include <iostream>

class Singleton {
 private:
  Singleton(const Singleton& old);  // disallow copy constructor
  const Singleton& operator=(
      const Singleton& old);  // disallow assignment operator

  /* Here will be the instance stored. */
  static Singleton* instance;

  /* Private constructor to prevent instancing. */
  Singleton();

 public:
  /* Static access method. */
  static Singleton* GetInstance();
};

/* Null, because instance will be initialized on demand. */
Singleton* Singleton::instance = 0;

Singleton* Singleton::GetInstance() {
  if (instance == 0) {
    instance = new Singleton();
  }

  return instance;
}

Singleton::Singleton() {}

int main() {
  // new Singleton(); // Won't work
  Singleton* s = Singleton::GetInstance();  // Ok
  Singleton* r = Singleton::GetInstance();

  /* The addresses will be the same. */
  std::cout << s << std::endl;
  std::cout << r << std::endl;
}
