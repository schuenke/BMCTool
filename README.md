# Bloch-McConnell (BMC) Simulation Tool

This repository contains a purely python-based Bloch-McConnell (BMC) simulation tool that can be used to simulate 
the evolution of the magnetization in various (exchanging) magnetic environments ('pools') under arbitrary
radio-frequency (RF) irradiation. The tool was developed to simulate Chemical Exchange Saturation Transfer (CEST) or 
related spectra, but can be used for many other MR simulations as well. 

The BMCTool utilizes the [pulseq](https://pulseq.github.io/) open file format to define and store all events (RF pulses,
gradients, delays, ADCs) that shall be simulated. The scanner settings and characteristic properties of the magnetic 
environments (relaxation times, pool size fractions, exchange rates) are defined and stored in config-files in the
[YAML](https://yaml.org) file format. 

Every simulation requires exactly one seq-file (containing all events) and at least one config-file. 

## Installation
The BMCTool can be installed from [PyPi](https://pypi.org/) using

``
pip install bmctool
``

Please keep in mind that the BMCTool is still in it's initial development and changes to the code might happen quite
often. Thus, for the moment, we recommend cloning the latest version from the 
[BMCTool GitHub repository](https://github.com/schuenke/BMCTool) using

``
git clone https://github.com/schuenke/BMCTool.git
``

and carefully tracking any changes/commits. 

### Initial Test
To make sure that the installation was successful, you can run an example simulation that is provided with both, 
the installation using pip and GitHub. To run the simulation, simply execute the [simulate.py](bmctool/simulate.py)
file or run the following code:
```python
from bmctool.simulate import simulate
simulate()
```
When no *.seq and *.yaml files are defined, the simulation uses the [WASABI.seq](bmctool/library/seq-library/WASABI.seq)
and [config_wasabi.yaml](bmctool/library/sim-library/config_wasabi.yaml) example files. The generated plot should look
like this:

![](https://raw.githubusercontent.com/schuenke/BMCTool/master/examples/example_wasabi_spectrum.png "Example WASABI spectrum")



## (CEST) Config and Seq-File Library
All simulations using the BMCTool (or the [pulseq-cest project](https://pulseq-cest.github.io/)) require a *yaml file* 
that includes all simulation settings and a *seq file*, which defines the events to be simulated. 
An [example seq-file](bmctool/library/seq-library/WASABI.seq) and an 
[example yaml file](bmctool/library/sim-library/config_wasabi.yaml) can be found in the [library](bmctool/library) 
subfolder. The [BMCTool GitHub repository](https://github.com/schuenke/BMCTool) further contains some example files to
create your own seq-files.

## Pulseq-CEST Project
The BMCTool was developed in parallel to the [pulseq-cest project](https://pulseq-cest.github.io/) that aims to provide
published and approved CEST saturation blocks in the [pulseq](https://pulseq.github.io/) open file format to enable an 
exact comparison of CEST saturation blocks with newly developed or adapted saturation blocks for reproducible research.
The [pulseq-cest project](https://pulseq-cest.github.io/) provides a [MATLAB implementation](https://github.com/kherz/pulseq-cest)
and a [python implementation](https://github.com/KerstinHut/pypulseq-cest) that both use the same Bloch-McConnell
equation solver implemented in C++. 

### Pulseq-CEST Library
You will find several pre-defined and approved CEST pre-saturation schemes and simulation configs in the 
[pulseq-cest-library](library/pulseq-cest-library) GitHub repository. You can clone the library using 

``
git clone https://github.com/kherz/pulseq-cest-library.git
``

or directly download the latest version as a [ZIP file](https://github.com/kherz/pulseq-cest-library/archive/master.zip).
