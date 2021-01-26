#!/usr/bin/env python
from distutils.core import setup
from distutils.command.clean import clean
from distutils.command.install import install

class performInstall(install):

    # Calls the default run command, then deletes the build area
    # (equivalent to "setup clean --all").
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()

if __name__ == '__main__':

    name_mdl = 'spectral'

    setup(
        name=name_mdl,
        version='1.0',
        author='Nick Karnesis',
        author_email='karnesis@apc.in2p3.fr',
        url='https://gitlab.in2p3.fr/Nikos/spectral',
        cmdclass={'install': performInstall},
        py_modules=[name_mdl]
        )

# END
