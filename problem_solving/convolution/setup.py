from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='cpu',
      ext_modules=[cpp_extension.CppExtension('cpu', ['cpu_conv.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
