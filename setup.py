#!/usr/bin/env python

import sys
if 'build_sphinx' in sys.argv or 'develop' in sys.argv:
    from setuptools import setup, Command
else:
    from distutils.core import setup, Command

with open('README.md') as file:
    long_description = file.read()

#with open('CHANGES') as file:
#    long_description += file.read()

# no versions yet from rampsclean import __version__ as version

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(name='rampsclean',
      version='0.1',
      description='Clean RAMPS spectra',
      long_description=long_description,
      author='Jonathan Foster',
      author_email='jonathan.b.foster@yale.edu',
      url='https://github.com/jfoster17/rampsclean',
      packages=['rampsclean',],
      cmdclass = {'test': PyTest},
     )