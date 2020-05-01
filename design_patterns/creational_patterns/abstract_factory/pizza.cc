/* clang++ pizza.cpp --std=c++14 */
/* use make_unique function with c++14 extension */

#include <iostream>
#include <memory>

class Pizza {
 public:
  virtual int getPrice() const = 0;
  virtual ~Pizza(){}; /* without this, no destructor for derived Pizza's will be
                         called. */
};

class MushroomPizza : public Pizza {
 public:
  virtual int getPrice() const { return 850; };
  virtual ~MushroomPizza(){};
};

class DeluxePizza : public Pizza {
 public:
  virtual int getPrice() const { return 1050; };
  virtual ~DeluxePizza(){};
};

class HawaiianPizza : public Pizza {
 public:
  virtual int getPrice() const { return 1150; };
  virtual ~HawaiianPizza(){};
};

class PizzaFactory {
 public:
  enum PizzaType {
    Mushroom,
    Deluxe,
    Hawaiian,
  };

  static std::unique_ptr<Pizza> createPizza(PizzaType pizzaType) {
    switch (pizzaType) {
      case Mushroom:
        return std::make_unique<MushroomPizza>();
      case Deluxe:
        return std::make_unique<DeluxePizza>();
      case Hawaiian:
        return std::make_unique<HawaiianPizza>();
    }
    throw "invalid pizza type.";
    return nullptr;
  }
};

/*
 * Create all available pizzas and print their prices
 */
void pizza_information(PizzaFactory::PizzaType pizzatype) {
  std::unique_ptr<Pizza> pizza = PizzaFactory::createPizza(pizzatype);
  std::cout << "Price of " << pizzatype << " is " << pizza->getPrice()
            << std::endl;
}

int main(void) {
  pizza_information(PizzaFactory::Mushroom);
  pizza_information(PizzaFactory::Deluxe);
  pizza_information(PizzaFactory::Hawaiian);

  return 0;
}
