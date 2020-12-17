from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bmctool',
    author='Patrick Schuenke',
    author_email='patrick.schuenke@ptb.de',
    version='0.1.0',
    description='A python tool perform Bloch-McConnell (BMC) simulations.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/schuenke/BMCTool',
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'PyYAML',
        'pypulseq @ git+https://github.com/imr-framework/pypulseq.git@dev'
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
    python_requires='>=3.7'
)
