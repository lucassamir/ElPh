import re
from pathlib import Path
from setuptools import setup, find_packages

txt = Path('elph/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)

long_description = Path('README.md').read_text()

setup(
    name='elph',
    version=version,
    description='Electron Phonon coupling within the Transient Localization Theory',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    author='Lucas Cavalcante',
    author_email='lsrcavalcante@ucdavis.edu',
    url='https://github.com/lucassamir/ElPh',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'ase>=3.20.0', 'tqdm', 'halo', 'phonopy', 'sphinx_rtd_theme', 'cmcrameri'],
    extras_require={'docs': ['sphinx', 'sphinxcontrib-programoutput']},
    entry_points='''
        [console_scripts]
        elph=elph.main:main
    ''',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: '
        'GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics'
    ])
