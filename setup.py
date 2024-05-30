import setuptools

setuptools.setup(
    name='data-scaling',
    version='0.0.1',
    author='Ian Covert',
    author_email='icovert@stanford.edu',
    description='For fitting individualized data scaling laws',
    url='https://github.com/iancovert/data-scaling',
    packages=['data_scaling'],
    install_requires=[
        # Basics.
        'tqdm',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        # ML.
        'scikit-learn',
        'torch==2.1.2',
        'torchvision==0.16.2',
        'timm>=0.9.10',
        'lightning==2.0.9',
        'opendataval==1.2.1',  # opendataval was installed from source, fixed some version compatibility issues
        'transformers==4.33.3',
        # Systems.
        'daal4py==2024.1.0',
        'scikit-learn-intelex==2024.1.0',
    ],
    python_requires='>=3.8',
)
