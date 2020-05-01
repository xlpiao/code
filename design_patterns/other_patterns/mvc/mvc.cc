// mvc.cpp
// https://helloacm.com/model-view-controller-explained-in-c/
#include <iostream>
#include <string>

#include "controller.h"
#include "model.h"
#include "view.h"

int main() {
  Model model("data.json");
  // register the data-change event

  View view(model);
  // binds model and view.
  Controller controller(model, view);
  // when application starts or button is clicked or form is shown...
  controller.OnRender();

  // this should trigger View to render
  model.SetData("update data.json");
  view.SetModel(model);
  controller.SetModel(model);
  controller.SetView(view);
  controller.OnRender();
  return 0;
}
