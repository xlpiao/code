#include <iostream>
#include <string>

class Door {
 public:
  virtual void Open() = 0;
  virtual void Close() = 0;
};

class RealDoor : public Door {
 public:
  void Open() override { std::cout << "Opening lab door" << std::endl; }
  void Close() override { std::cout << "Closing the lab door" << std::endl; }
};

class ProxyDoor {
 public:
  ProxyDoor(Door& door) : door_(door) {}
  bool Authenticate(const std::string& password) {
    return password == "$ecr@t";
  }
  void Open(const std::string& password) {
    if (Authenticate(password))
      door_.Open();
    else
      std::cout << "Big no! It ain't possible." << std::endl;
  }
  void Close() { door_.Close(); }

 private:
  Door& door_;
};

int main() {
  RealDoor realDoor;
  ProxyDoor proxySecureDoor(realDoor);
  proxySecureDoor.Open("invalid");
  proxySecureDoor.Open("$ecr@t");
  proxySecureDoor.Close();
}
