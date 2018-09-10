#include <iostream>

class AAA {
  public:
    void String() {
        std::cout << "AAA::String" << std::endl;
    }
};

class BBB {
  public:
    void String() {
        std::cout << "BBB::String" << std::endl;
    }
};

class CCC : public AAA, public BBB {
  public:
    void ShowString() {
        // String();
        // String();
        AAA::String();
        BBB::String();
    }
};

int main(void) {
    CCC ccc;
    ccc.ShowString();

    return 0;
}
