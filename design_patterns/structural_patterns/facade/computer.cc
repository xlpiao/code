#include <iostream>

class Computer {
 public:
  void GetElectricShock() {
    std::cout << "1. GetElectricShock: Ouch!" << std::endl;
  }
  void MakeSound() { std::cout << "2. MakeSound: Beep beep!" << std::endl; }
  void ShowLoadingScreen() {
    std::cout << "3. ShowLoadingScreen: Loading..." << std::endl;
  }
  void Bam() { std::cout << "4. Bam: Ready to be used!" << std::endl; }
  void CloseEverything() {
    std::cout << "5. CloseEverything: Bup bup bup buzzzz!" << std::endl;
  }
  void PullCurrent() { std::cout << "6. PullCurrent: Haaah!" << std::endl; }
  void Sooth() { std::cout << "7. Sooth: Zzzzz" << std::endl; }
};

class ComputerFacade {
 public:
  ComputerFacade(Computer& computer) : computer_(computer) {}
  void TurnOn() {
    computer_.GetElectricShock();
    computer_.MakeSound();
    computer_.ShowLoadingScreen();
    computer_.Bam();
  }
  void TurnOff() {
    computer_.CloseEverything();
    computer_.PullCurrent();
    computer_.Sooth();
  }

 private:
  Computer& computer_;
};

int main() {
  Computer real_computer;
  ComputerFacade computer(real_computer);
  computer.TurnOn();
  computer.TurnOff();
}
