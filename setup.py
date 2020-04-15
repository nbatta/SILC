from distutils.core import setup, Extension
import os

setup(name='SILC',
      version='0.1',
      description='Simple Internal Linear Combination Code for Power Spectrum Forecasts',
      url='https://github.com/nbatta/SILC',
      author='Nicholas Battaglia and Yijie Zhu',
      author_email='nicholas.battaglia@gmail.com',
      license='BSD-2-Clause',
      packages=['SILC'],
      package_dir={'SILC':'silc'},
      zip_safe=False)
