// view.h
// https://helloacm.com/model-view-controller-explained-in-c/
#pragma once
#include <iostream>

#include "model.h"
// View is responsible to present data to users
class View {
 public:
  View(const Model &model) {
    std::cout << "View: View("
              << "Model"
              << ")" << std::endl;
    this->model = model;
  }
  View() {}
  void SetModel(const Model &model) {
    std::cout << "View: setModel()" << std::endl;
    this->model = model;
  }
  void Render() {
    std::cout << "View: Render(" << model.Data() << ")\n" << std::endl;
  }

 private:
  Model model;
};
