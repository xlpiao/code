#include <cstdlib>
#include <iostream>

class Boy {
 public:
  Boy(int len) : height(len) {}

 private:
  int height;
  friend class Girl;
};

class Girl {
 public:
  Girl(int len) : height(len) {}
  void ShowInfo() { std::cout << "Girl's height: " << height << std::endl; }
  void ShowInfo(Boy &boy) {
    std::cout << "Boy's height: " << boy.height << std::endl;
  }

 private:
  int height;
};

int main(void) {
  Boy boy(180);
  Girl girl(160);

  girl.ShowInfo();
  girl.ShowInfo(boy);

  return 0;
}
