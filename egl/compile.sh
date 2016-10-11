cc -g `pkg-config --cflags gbm` `pkg-config --cflags egl` `pkg-config --cflags gl` -std=c99 -o output egl_gbm.c `pkg-config --libs gbm`  `pkg-config --libs egl` `pkg-config --libs gl`
