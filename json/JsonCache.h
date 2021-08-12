#pragma once

#ifdef _WIN32
#include <json/json.h>
#else
#include <jsoncpp/json/json.h>
#endif

#include <string>

namespace jaws {

namespace key {
const std::string url{"url"};
const std::string host{"host"};
const std::string port{"port"};
}  // namespace key

class JsonCache {
public:
  explicit JsonCache(const std::string &filePath);
  ~JsonCache();

  std::string readAsString();
  Json::Value readAsObject();
  void write();

  void setHost(const std::string &host);
  void setPort(const std::string &port);
  const std::string getHost();
  const std::string getPort();
  void setUrl(const std::string &url);

private:
  std::string filePath_;
  Json::Value data_;
};
}  // namespace jaws
