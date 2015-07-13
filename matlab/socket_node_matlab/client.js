net = require('net');
HOST='localhost';
PORT=5000;

client = new net.Socket();

client.connect(PORT, HOST, function() {
    console.log('Connected');
});

client.on('data', function(data) {
    console.log('Received: ' + data);
});

client.on('close', function() {
    console.log('Connection closed');
    client.destroy();
});