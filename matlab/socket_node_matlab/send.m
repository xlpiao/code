function send(socket, data)
    s = whos(data);
    fwrite(socket, data, s.class);
end
