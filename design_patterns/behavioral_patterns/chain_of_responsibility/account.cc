#include <exception>
#include <iostream>
#include <string>

class Account {
 public:
  Account(float balance) : balance_(balance) {
    std::cout << "Account: " << balance_ << std::endl;
  }
  virtual std::string GetClassName() { return "Account"; }
  void SetNext(Account* const account) { successor_ = account; }
  bool CanPay(float amount) { return balance_ >= amount; }
  void Pay(float amountToPay) {
    if (CanPay(amountToPay)) {
      std::cout << "Pay using " << GetClassName() << ". Proceeding..."
                << std::endl;
      std::cout << "Account balance: " << balance_ << ", Paid " << amountToPay
                << " using " << GetClassName() << std::endl;
    } else if (successor_) {
      std::cout << "Cannot pay using " << GetClassName() << ". Proceeding..."
                << std::endl;
      successor_->Pay(amountToPay);
    } else {
      throw "None of the accounts have enough balance.";
    }
  }

 protected:
  Account* successor_ = nullptr;
  float balance_;
};

class Bank : public Account {
 public:
  Bank(float balance) : Account(balance) {
    std::cout << "Bank Accout: " << balance << std::endl;
  }
  std::string GetClassName() override { return "Bank"; }
};

class Paypal : public Account {
 public:
  Paypal(float balance) : Account(balance) {
    std::cout << "Paypal Accout: " << balance << std::endl;
  }
  std::string GetClassName() override { return "Paypal"; }
};

class Bitcoin : public Account {
 public:
  Bitcoin(float balance) : Account(balance) {
    std::cout << "Bitcoin Accout: " << balance << std::endl;
  }
  std::string GetClassName() override { return "Bitcoin"; }
};

int main() {
  //! Let's prepare a chain like below:
  //! bank -> paypal -> bitcoin
  //! First priority bank
  //!   If bank can't pay then paypal
  //!   If paypal can't pay then bit coin

  Bank bank(100);        //> Bank with balance 100
  Paypal paypal(200);    //> Paypal with balance 200
  Bitcoin bitcoin(300);  //> Bitcoin with balance 300

  bank.SetNext(&paypal);
  paypal.SetNext(&bitcoin);

  bank.Pay(59);
  bank.Pay(159);
  bank.Pay(259);
}
