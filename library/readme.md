# Config and seq-file library
This folder contains one [config file](config_example.yaml), one [(py)pulseq-cest seq-file](seq_example.seq) and the
corresponding [python file](write_seq_example.py) to create the seq-file. More information about the pulseq-cest-library
folder can be found below.

## pulseq-cest-library
The [pulseq-cest-library](https://github.com/kherz/pulseq-cest-library) is a separate GitHub repository that contains 
several pre-defined and approved pre-saturation schemes 
([seq-library](https://github.com/kherz/pulseq-cest-library/tree/master/seq-library)) as well as several pre-defined 
and approved simulation settings ([sim-library](https://github.com/kherz/pulseq-cest-library/tree/master/sim-library)).
More information about the structure and the content of these files can be found in the corresponding 
[seq-library readme](https://github.com/kherz/pulseq-cest-library/blob/master/seq-library/Readme.md) and 
[sim-library readme](https://github.com/kherz/pulseq-cest-library/blob/master/sim-library/Readme.md) files.

### How to dowload the pulse-cest-library in case it's empty
The [pulseq-cest-library](https://github.com/kherz/pulseq-cest-library) is uncluded as a 
[GitHub submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules). If your 
[pulseq-cest-library](pulseq-cest-library) folder ist empty, you didn't clone the the pypulseq-cest repository including
submodules (as explained in the [pypulseq-cest readme](../readme.md)) or you downloaded the [pypulseq-cest repository]() 
as a ZIP file. However, you can still get the files using one of these two options:

**Option 1:** Use the following GitHub commands (in the [pypulseq-cest]() directory) to initialize your local configuration file and to fetch all data:
```
git submodule init
git submodule update
``` 
**Option 2:** Download the [pulseq-cest-library](https://github.com/kherz/pulseq-cest-library) as a 
[ZIP file](https://github.com/kherz/pulseq-cest-library/archive/master.zip) and extract it into the 
[pulseq-cest-library](pulseq-cest-library).

### Update the pulseq-cest-library
If the pulseq-cest-library was updated and you want to pull these updates, use the following command:
```
git submodule update --remote
``` 
