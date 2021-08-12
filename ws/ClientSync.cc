#include "ClientSync.h"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <string>

namespace ws {

ClientSync::ClientSync() : ws_{ioc_} {}

ClientSync::~ClientSync() {
  ws_.close(boost::beast::websocket::close_code::normal);
}

void ClientSync::connect(const std::string& host, const std::string& port) {
  boost::asio::ip::tcp::resolver resolver{ioc_};
  auto const results = resolver.resolve(host, port);
  boost::asio::connect(ws_.next_layer(), results);
  ws_.handshake(host + ':' + port, "/websocket");
  if (ip_.empty()) {
    ip_ = ws_.next_layer().local_endpoint().address().to_string();
  }
}

const std::string ClientSync::getIp() { return ip_; }

void ClientSync::send(const std::string& message) {
  ws_.write(boost::asio::buffer(message));
}

void ClientSync::recv() {
  boost::beast::flat_buffer buffer;
  ws_.read(buffer);
  std::cout << boost::beast::make_printable(buffer.data()) << std::endl;
}
bool ClientSync::isOpen() { return ws_.is_open(); }
}  // namespace ws
