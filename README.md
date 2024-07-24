# Bloch-McConnell (BMC) Simulation Tool

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Coverage Bagde](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/schuenke/6296bce292048be38f06cc216719f558/raw/coverage.json)

This repository contains a purely Python-based Bloch-McConnell (BMC) simulation tool that can be used to simulate the evolution of magnetization in various (exchanging) magnetic environments ('pools') with arbitrary radio-frequency (RF) irradiation schemes. The tool was developed to simulate Chemical Exchange Saturation Transfer (CEST) or related spectra but can be used for many other MR simulations as well.

The BMCTool utilizes the [pulseq](https://pulseq.github.io/) open file format to define and store all events (RF pulses, gradients, delays, ADCs) to be simulated. The scanner settings (e.g., field strength, B0 inhomogeneity) and characteristic properties of the magnetic environments (e.g., relaxation times, pool size fractions, exchange rates) are defined and stored in configuration files in the [YAML](https://yaml.org) file format.

Every simulation requires exactly one Pulseq sequence file (containing all events) and at least one configuration file.

## Installation

The BMCTool is available on [PyPi](https://pypi.org/) and can be installed using `pip install bmctool`.

Alternatively, if Git is available on your system, you can directly install BMCTool from GitHub using `pip install git+https://github.com/schuenke/bmctool@main`. You can change the `@main` to any desired branch.

### Initial Test

To make sure that the installation was successful, you can run an example simulation that is provided with both, the installation using pip and GitHub. To run an example simulation, simply execute the following code:

```python
from bmctool.simulation import sim_example

sim_example()
```

The sim_example function uses the [WASABI.seq](src/bmctool/library/seq-library/WASABI.seq) and [config_1pool.yaml](src/bmctool/library/sim-library/config_1pool.yaml) example files.

## Running a Simulation

All simulations using the BMCTool require at least one config file (in the *YAML* format) that includes all simulation settings and exactly one Pulseq sequence file (in the *seq* format), which defines the events to be simulated. An [example seq-file](src/bmctool/library/seq-library/WASABI.seq) and an [example yaml file](src/bmctool/library/sim-library/config_1pool.yaml) can be found in the [library](src/bmctool/library) subfolder. For more information about configuration files and sequence files as well as about the [Pulseq-CEST library](https://github.com/kherz/pulseq-cest-library), where both types of files can be shared, please read the [Pulseq-CEST Library](#pulseq-cest-library) section below.

If you created your own files or downloaded them from the [pulseq-cest-library](https://github.com/kherz/pulseq-cest-library), you can start the simulation by running the following code:

```python
from bmctool.simulate import simulate

config_path = '<path_to_your_config>'  # can be a str or a Path
seq_path = '<path_to_your_sequence>'  # can be a str or a Path
sim = simulate(config_file=config_path, seq_file=seq_path, show_plot=True)
```

The simulate function accepts several additional keyword arguments (`**kwargs`), that allow to adjust the plot. These are for example ***normalize*** (bool: toggle normalization), ***norm_threshold*** (value/list/array: threshold for normalization offsets), ***offsets*** (list/array: manually defined x-values), ***invert_ax*** (bool: toggle invert ax), ***plot_mtr_asym*** (bool:toggle plot MTR_asym) and ***title***, ***x_label***, ***y_label*** to control the labels.

## Pulseq-CEST Project

The BMCTool was developed in parallel to the [pulseq-cest project](https://pulseq-cest.github.io/) that aims to provide published and approved CEST saturation blocks in the [pulseq](https://pulseq.github.io/) open file format to enable an exact comparison of CEST saturation blocks with newly developed or adapted saturation blocks for reproducible research. The [pulseq-cest project](https://pulseq-cest.github.io/) provides a [MATLAB implementation](https://github.com/kherz/pulseq-cest) as well.

### Pulseq-CEST Library

You will find several pre-defined and approved CEST pre-saturation schemes and simulation configs in the [Pulseq-CEST-library on GitHub](https://github.com/kherz/pulseq-cest-library). You can clone the library using `git clone https://github.com/kherz/pulseq-cest-library.git` or download the latest version as a [ZIP file](https://github.com/kherz/pulseq-cest-library/archive/master.zip).
