#include <iostream>
using namespace std;

class AAA {
 public:
  AAA() { cout << "AAA()" << endl; }
  ~AAA() { cout << "~AAA()" << endl; }

  void ShowYou() { cout << "AAA exception!" << endl; }
};

class BBB : public AAA {
 public:
  BBB() { cout << "BBB()" << endl; }
  ~BBB() { cout << "~BBB()" << endl; }

  void ShowYou() { cout << "BBB exception!" << endl; }
};

class CCC : public BBB {
 public:
  CCC() { cout << "CCC()" << endl; }
  ~CCC() { cout << "~CCC()" << endl; }

  void ShowYou() { cout << "CCC exception!" << endl; }
};

void ExceptionGenerator(int expn) {
  if (expn == 1)
    throw AAA();
  else if (expn == 2)
    throw BBB();
  else
    throw CCC();
}

int main(void) {
  try {
    ExceptionGenerator(1);
  } catch (CCC& expn) {
    cout << "catch(CCC& expn)" << endl;
    expn.ShowYou();
  } catch (BBB& expn) {
    cout << "catch(BBB& expn)" << endl;
    expn.ShowYou();
  } catch (AAA& expn) {
    cout << "catch(AAA& expn)" << endl;
    expn.ShowYou();
  }
  return 0;
}
