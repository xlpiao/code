net = require('net');
queue = require('./Queue');

function Client(host, port){
    this.Q = new queue();
    this.socket = new net.Socket();
    this.socket.connect(port, host, function() {
        console.log('Connected');
    });
}

Client.prototype.send = function (data){
    this.socket.write(data+'\n');
}

Client.prototype.receive = function (){
    var that = this;
    this.socket.on('data', function(data) {
        that.Q.enqueue(data);
        console.log(''+data);
    });
}

Client.prototype.disconnect = function (){
    this.socket.on('close', function() {
        console.log('Connection closed');
        this.socket.destroy();
    });
}

module.exports = Client;
