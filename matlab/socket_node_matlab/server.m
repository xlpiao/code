%%
%% server.m
%% Copyright (c) 2016 Xianglan Piao <lanxlpiao@gmail.com>
%%
%% Licensed under the Apache License, Version 2.0 (the "License");
%% you may not use this file except in compliance with the License.
%% You may obtain a copy of the License at
%%
%%     http://www.apache.org/licenses/LICENSE-2.0
%%
%% Unless required by applicable law or agreed to in writing, software
%% distributed under the License is distributed on an "AS IS" BASIS,
%% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%% See the License for the specific language governing permissions and
%% limitations under the License.
%%
%%
%% @file server.m
%% @author Xianglan Piao <lanxlpiao@gmail.com>
%% Date: 2016.10.07
%%

function s = server(host, port)
    s.Q = Queue();
    s.socket = tcpip(host,port,'NetworkRole','Server');
    set(s.socket, 'InputBufferSize', 3000000);
    set(s.socket, 'OutputBufferSize', 3000000);
    fopen(s.socket);

    s.send = @send;
    function send(data)
        d = whos(data);
        fwrite(s.socket, data, d.class);
    end

    s.receive = @receive;
    function receive()
        TimerFcn = {@recv, s.socket, s.Q};
        t = timer('ExecutionMode', 'FixedRate', ...
            'Period', 1, ...
            'TimerFcn', TimerFcn);
        start(t);
    end

    s.disconnect = @disconnect;
    function disconnect()
        fclose(s.socket);
        delete(s.socket);
        clear s.socket;
    end
end

function Q = recv(hobj, eventdata, socket, Q)
    while(socket.BytesAvailable > 0)
        data = fgetl(socket);
        Q.enqueue(data);
        fprintf('%s\n', data);
    end
end
