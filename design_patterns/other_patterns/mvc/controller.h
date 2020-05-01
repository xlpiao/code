// controller.h
// https://helloacm.com/model-view-controller-explained-in-c/
#pragma once
#include "model.h"
#include "view.h"

// Controller combines Model and View
class Controller {
 public:
  Controller(const Model &model, const View &view) {
    this->SetModel(model);
    this->SetView(view);
  }
  void SetModel(const Model &model) {
    this->model = model;
    std::cout << "Controller: SetModel()" << std::endl;
  }
  void SetView(const View &view) {
    std::cout << "Controller: SetView()" << std::endl;
    this->view = view;
  }
  // when application starts
  void OnRender() {
    std::cout << "Controller: OnRender() calls view.Render()" << std::endl;
    this->view.Render();
  }

 private:
  Model model;
  View view;
};
