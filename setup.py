from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='BMCTool',
    author='Patrick Schuenke',
    author_email='patrick.schuenke@ptb.de',
    version='0.3.2',
    description='A python tool to perform Bloch-McConnell (BMC) simulations.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/schuenke/BMCTool',
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'PyYAML',
        'deprecation',
        'pypulseq>=1.3.1post1',
    ],
    keywords='MRI, Bloch, CEST, simulations',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)
