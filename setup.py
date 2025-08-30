from setuptools import setup, find_packages

setup(
    name='densenet_pytorch',
    version='0.1.0',
    author='Joshua Nikiema',
    description='A PyTorch implementation of Densely Connected Convolutional Networks (DenseNet)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Yunpei24/DenseNet_package_from_jyen', # Change to your repo URL
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.11',
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pandas>=1.0.0',
        'tqdm>=4.0.0',
        'wandb>=0.12.0',
    ],
)