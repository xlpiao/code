// model.h
// https://helloacm.com/model-view-controller-explained-in-c/
#pragma once
#include <string>

typedef void (*DataChangeHandler)(std::string newData);

void DataChange(std::string data) {
  std::cout << "Model: Data Changes: " << data << std::endl;
}

class Model {
 public:
  Model(const std::string &data) {
    this->SetData(data);
    this->RegisterDataChangeHandler(&DataChange);
  }
  Model() {}  // default constructor
  std::string Data() { return this->data; }

  void SetData(const std::string &data) {
    std::cout << "Model: " << data << std::endl;
    this->data = data;
    if (this->event != nullptr) {  // data change callback event
      this->event(data);
    }
  }

  // register the event when data changes.
  void RegisterDataChangeHandler(DataChangeHandler handler) {
    std::cout << "Model: RegisterDataChangeHandler()" << std::endl;
    this->event = handler;
  }

 private:
  std::string data = "";
  DataChangeHandler event = nullptr;
};
