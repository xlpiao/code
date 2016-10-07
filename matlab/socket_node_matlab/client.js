/*
 * client.js
 * Copyright (c) 2016 Xianglan Piao <xianglan0502@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * @file client.js
 * @author Xianglan Piao <xianglan0502@gmail.com>
 * Date: 2016.10.07
 */
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
