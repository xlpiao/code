from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cpu',
      ext_modules=[cpp_extension.CppExtension('cpu', ['cpu_conv2d.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='gpu',
      ext_modules=[cpp_extension.CppExtension('gpu', ['gpu_conv2d.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
