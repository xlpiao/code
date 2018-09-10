#include <cstdlib>
#include <iostream>

class Boy {
  private:
    int height;
    friend class Girl;

  public:
    Boy(int len) : height(len) {}
};

class Girl {
  private:
    int height;

  public:
    Girl(int len) : height(len) {}
    void ShowInfo() { std::cout << "Her height: " << height << std::endl; }
    void ShowYourFriendInfo(Boy &frn) {
        std::cout << "His height: " << frn.height << std::endl;
    }
};

int main(void) {
    Boy boy(180);

    Girl girl(160);
    girl.ShowInfo();
    girl.ShowYourFriendInfo(boy);

    return 0;
}
