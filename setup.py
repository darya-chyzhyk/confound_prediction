from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="confound_prediction",
    version="0.0.1a",
    author="Darya Chyzhyk",
    author_email="darya.chyzhyk@gmail.com",
    description="Confound-isolating cross-validation approach to control for a "
                "confounding effect in a predictive model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darya-chyzhyk/confound_prediction",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering',

    ],
    platforms='any',
    install_requires=['scipy>=1.1.0', 'scikit-learn>=0.21.2', 'numpy>=1.14.2']
)

