import re
from pathlib import Path
from setuptools import setup, find_packages

txt = Path('tloc/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)

long_description = Path('README.md').read_text()

setup(
    name='tloc',
    version=version,
    description='Transient Localization Theory',
    #long_description=long_description,
    #long_description_content_type='text/markdown',
    author='Lucas Cavalcante',
    author_email='lsrcavalcante@ucdavis.edu',
    url='https://github.com/lucassamir/TLoc',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'ase>=3.20.0', 'tqdm', 'halo'],
    extras_require={'docs': ['sphinx', 'sphinxcontrib-programoutput']},
    entry_points='''
        [console_scripts]
        tloc=tloc.tloc:main
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
