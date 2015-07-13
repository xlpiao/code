function receive(socket, Q)
    TimerFcn = {@recv, socket, Q};
    t = timer('ExecutionMode', 'FixedRate', ...
        'Period', 1, ...
        'TimerFcn', TimerFcn);
    start(t);
end

function Q = recv(hobj, eventdata, socket, Q)
    while(socket.BytesAvailable > 0)
        data = fgetl(socket);
        Q.enqueue(data);
        fprintf('%s\n', data);
    end
end
