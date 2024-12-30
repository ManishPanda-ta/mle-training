from setuptools import find_packages
from setuptools import setup

setup(
    name='python-pack-assign',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    # You can add more metadata here if needed
    author='Manish Panda',
    author_email='manish.panda@tigeranalytics.com',
    description='A short description for the package',
    url='https://github.com/ManishPanda-ta/mle-training/python-pack-assign',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
