#include <iostream>
using namespace std;

/* Place holder for thread synchronization mutex */
class Mutex { /* placeholder for code to create, use, and free a mutex */
};

/* Place holder for thread synchronization lock */
class Lock {
 public:
  Lock(Mutex& m) : mutex(m) { /* placeholder code to acquire the mutex */
  }
  ~Lock() { /* placeholder code to release the mutex */
  }

 private:
  Mutex& mutex;
};

class SingletonMutex {
 public:
  static SingletonMutex* GetInstance();
  int a;
  ~SingletonMutex() { cout << "In Destructor" << endl; }

 private:
  SingletonMutex(int _a) : a(_a) { cout << "In Constructor" << endl; }

  static Mutex mutex;

  // Not defined, to prevent copying
  SingletonMutex(const SingletonMutex&);
  SingletonMutex& operator=(const SingletonMutex& other);
};

Mutex SingletonMutex::mutex;

SingletonMutex* SingletonMutex::GetInstance() {
  Lock lock(mutex);

  cout << "Get Instance" << endl;

  // Initialized during first access
  static SingletonMutex inst(1);

  return &inst;
}

int main(void) {
  SingletonMutex* singleton_mutex = SingletonMutex::GetInstance();
  cout << "The value of the singleton_mutex: " << singleton_mutex->a << endl;
  return 0;
}
