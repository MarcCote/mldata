# -*- coding: utf-8 -*-
from distutils.core import setup

setup(name="mldata",
      version='0.0.0',
      author='Adam Salvail, Marc-Alexandre Côté',
      author_email='smart-udes-dev@googlegroups.com',
      packages=['mldata'],
      url='https://github.com/SMART-Lab/mldata',
      scripts=[],
      license='LICENSE.txt',

      description="A library to manage machine learning datasets.",
      long_description=open('README.txt').read(),

      install_requires=['numpy>=1.7.0', 'h5py'],
      tests_require=['nose'],
      test_suite="nose.collector",
      )
