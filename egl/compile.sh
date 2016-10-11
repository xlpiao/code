# export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/system/lib
# export C_INCLUDE_PATH=/usr/local/include:$C_INCLUDE_PATH

cc -g `pkg-config --cflags gbm` `pkg-config --cflags egl` `pkg-config --cflags gl` -std=c99 -o egl_gbm egl_gbm.c `pkg-config --libs gbm`  `pkg-config --libs egl` `pkg-config --libs gl`

cc -g `pkg-config --cflags gbm` `pkg-config --cflags egl` `pkg-config --cflags gl` -std=c99 -o egl_example egl_example.c `pkg-config --libs gbm`  `pkg-config --libs egl` `pkg-config --libs gl`
