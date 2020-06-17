import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import fftpack
import tensorflow as tf
from tensorflow import keras

## one dim data
x = np.zeros(500)
x[50:100] = 1
x[150:200] = 1
x[250:300] = 1
x[350:400] = 1
x[450:500] = 1

X = fftpack.fftshift(np.abs(fftpack.fft(x))) ## fft

plt.subplots(2, 1)
plt.subplot(2,1,1)
plt.plot(x[0:])
plt.subplot(2,1,2)
plt.plot(X)

## two dim data
y = np.zeros((500,500))
y[50:100, 50:100] = 32
y[150:200,150:200] = 64
y[250:300,250:300] = 128
y[350:400,350:400] = 192
y[450:500,450:500] = 256

Y = fftpack.fftshift(np.abs(fftpack.fft2(y)))

plt.subplots(2, 1)
plt.subplot(2,1,1)
plt.plot(y[0:,0:])
plt.subplot(2,1,2)
plt.plot(fft2)

plt.subplots(2, 1)
plt.subplot(2,1,1)
plt.imshow(y[0:,0:], interpolation='nearest')
plt.subplot(2,1,2)
plt.imshow(Y[0:,0:], interpolation='nearest')

## two dim data: image
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.subplots(2, 1)
plt.subplot(2,1,1)
plt.imshow(train_images[0], interpolation='nearest')
fft_train_images = fftpack.fftshift(np.abs(fftpack.fft2(train_images[0])))
plt.subplot(2,1,2)
plt.imshow(fft_train_images, interpolation='nearest')

## two dim data: nine images
nine_images = np.hstack((train_images[0], train_images[1], train_images[2]))
nine_images = np.vstack((nine_images, nine_images, nine_images))

plt.subplots(2, 1)
plt.subplot(2,1,1)
plt.imshow(nine_images, interpolation='nearest')
fft_nine_images = fftpack.fftshift(np.abs(fftpack.fft2(nine_images)))
plt.subplot(2,1,2)
plt.imshow(fft_nine_images, interpolation='nearest')

plt.show()
