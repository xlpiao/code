/* clang++ computer.cpp --std=c++11 */
/* use 'override' keyword with C++11 extension */

#include <iostream>

class Computer {
 public:
  virtual void Run() = 0;
  virtual void Stop() = 0;

  virtual ~Computer(){}; /* without this, you do not call Laptop or Desktop
                            destructor in this example! */
};
class Laptop : public Computer {
 public:
  void Run() override { mHibernating = false; };
  void Stop() override { mHibernating = true; };
  virtual ~Laptop(){}; /* because we have virtual functions, we need virtual
                          destructor */
 private:
  bool mHibernating;  // Whether or not the machine is hibernating
};
class Desktop : public Computer {
 public:
  void Run() override { mOn = true; };
  void Stop() override { mOn = false; };
  virtual ~Desktop(){};

 private:
  bool mOn;  // Whether or not the machine has been turned on
};

class ComputerFactory {
 public:
  static Computer *NewComputer(const std::string &description) {
    if (description == "laptop") {
      std::cout << "This is laptop computer" << std::endl;
      return new Laptop;
    }
    if (description == "desktop") {
      std::cout << "This is desktop computer" << std::endl;
      return new Desktop;
    }
    return nullptr;
  }
};

int main(void) {
  Computer *laptop = ComputerFactory::NewComputer("laptop");
  Computer *desktop = ComputerFactory::NewComputer("desktop");
  Computer *dummy = ComputerFactory::NewComputer("dummy");

  return 0;
}
