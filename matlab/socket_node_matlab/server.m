function [Q socket] = server(host, port)
    Q = Queue();
    socket = tcpip(host,port,'NetworkRole','Server');
    set(socket, 'InputBufferSize', 3000000);
    set(socket, 'OutputBufferSize', 3000000);
    fopen(socket);
end
