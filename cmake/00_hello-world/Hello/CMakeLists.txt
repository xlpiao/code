add_library (Hello 
             Speaker.h
             Speaker.cpp)

install (TARGETS Hello DESTINATION /usr/bin)
install (FILES Speaker.h DESTINATION /usr/include)
