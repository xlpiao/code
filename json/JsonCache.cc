#include "JsonCache.h"

#include <boost/regex.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace jaws {

JsonCache::JsonCache(const std::string &filePath) : filePath_{filePath} {
  auto realPath = std::filesystem::path(filePath_);
  auto dir = realPath.parent_path().u8string();
  auto name = realPath.filename().u8string();
  // std::cout << dir << ", " << name << std::endl;

  if (!dir.empty() && !(std::filesystem::exists(dir))) {
    if (!std::filesystem::create_directories(dir)) {
      std::cout << "[JsonCache] " << dir << "create failed" << std::endl;
    }
  }

  if (!std::filesystem::exists(realPath)) {
    std::ofstream jsonFile(filePath_);
    jsonFile.close();
  }

  data_ = readAsObject();
  write();
}

JsonCache::~JsonCache() {}

Json::Value JsonCache::readAsObject() {
  Json::Value data;
  Json::CharReaderBuilder builder;

  std::ifstream file(filePath_);
  if (file.good()) {
    Json::parseFromStream(builder, file, &data, nullptr);
  }
  file.close();
  return data;
}

std::string JsonCache::readAsString() {
  Json::StreamWriterBuilder builder;
  std::string dataStr = Json::writeString(builder, data_);
  return dataStr;
}

void JsonCache::write() {
  std::ofstream jsonFile(filePath_);
  Json::StreamWriterBuilder builder;
  builder.newStreamWriter()->write(data_, &jsonFile);
  jsonFile.close();
}

void JsonCache::setHost(const std::string &host) {
  data_[key::url][key::host] = host;
}

const std::string JsonCache::getHost() {
  auto host = data_[key::url][key::host].asString();
  std::cout << "host: " << host << std::endl;
  return host;
}

void JsonCache::setPort(const std::string &port) {
  data_[key::url][key::port] = port;
}

const std::string JsonCache::getPort() {
  auto port = data_[key::url][key::port].asString();
  std::cout << "port: " << port << std::endl;
  return port;
}

void JsonCache::setUrl(const std::string &url) {
  std::string host;
  std::string port;

  boost::regex ex(
      "(http|https)://([^/ :]+):?([^/ ]*)(/?[^ #?]*)\\x3f?([^ #]*)#?([^ ]*)");
  boost::cmatch what;
  if (regex_match(url.c_str(), what, ex)) {
    host = std::string(what[2].first, what[2].second);
    port = std::string(what[3].first, what[3].second);
    if (port.empty()) {
      port = "80";
    }
  }

  setHost(host);
  setPort(port);
}

}  // namespace jaws
