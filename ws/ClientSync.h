#include <boost/asio.hpp>
#include <boost/beast/websocket.hpp>
#include <string>

namespace ws {

class ClientSync {
public:
  ClientSync();
  ~ClientSync();

  void connect(const std::string& host, const std::string& port);
  void send(const std::string& message);
  void recv();
  bool isOpen();

  const std::string getIp();

private:
  boost::asio::io_context ioc_;
  boost::beast::websocket::stream<boost::asio::ip::tcp::socket> ws_;
  std::string ip_;
};

}  // namespace ws
