"""Env setup.py."""
import os
import re

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension


def parse_requirements(requie_file='requirements.txt'):
  assert os.path.exists(requie_file), f"{requie_file} doesn't exists!"
  with open(requie_file, 'r') as f:
    data = [
        line.strip('\n')
        for line in f.readlines()
        if not re.match(r'^\s*(-i|--extra-index-url).*', line)
    ]
  return data


if __name__ == '__main__':
  setup(
      name='mapsa',
      version='0.0.0',
      description='Multimodal Aspect-Predicted Sentiment Analysis',
      author='Voyagers',
      author_email='h.wolf@qq.com',
      url='https://github.com/MEIMEIMEIMEIMEMEDA/MAPSA',
      packages=['mapsa'],
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      # namespace_packages=['mapsa'],
      setup_requires=parse_requirements('requirements.txt'),
      cmdclass={'build_ext': BuildExtension},
  )
