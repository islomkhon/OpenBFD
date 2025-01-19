from setuptools import setup, find_packages

setup(
    name='openbfd',  # Your project name
    version='0.1',  # Version of your project
    packages=find_packages(
        where='.',  # Start from the root directory
        include=['DLModels', 'FitLoops', 'Datasets', 'Notebook']),  # Automatically find packages in the project
    install_requires=[  # List your dependencies here
        'torch',  
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
    ],
    # entry_points={  # Define an entry point (if you have a CLI)
    #     'console_scripts': [
    #         'my_project=main:mai
    # n',  # This will call the main function in main.py
    #     ],
    # },
    author='Islomkhon Nizomkhonov',  # Author name
    author_email='islomkhon@mail.com',  # Author's email
    description='OpenBFD is an open-source deep learning tool for bearing fault diagnosis built with PyTorch. It provides comprehensive solutions for detecting and classifying bearing faults using raw vibration data and advanced deep learning techniques.',  # Short description of the project
    long_description=open('README.md', encoding='utf-8').read(),  # Read the README file as the long description
    long_description_content_type='text/markdown',  # Markdown for long description
    url='https://github.com/islomkhon/OpenBFD',  # URL for your project (e.g., GitHub)
    classifiers=[  # Classifiers for PyPI (optional, but recommended)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8.10',  # Python version requirement
)
