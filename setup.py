#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(name='deep-belief-network',
      version='1.0.1',
      description='Python implementation of Deep Belief Networks',
      packages=['dbn', 'dbn.tensorflow'],
      install_requires=requirements,
      )

setup(name='stacked-autoencoder',
        version='1.0.1',
        description='Python implementation of Stacked Autoencoder',
        packages=['sae'],
        install_requires=requirements,
        )