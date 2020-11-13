## Sequence design in open file format for MR sequences
This repository contains the necessary code and tools to build CEST saturation blocks with a variation of [pypulseq](https://github.com/imr-framework/pypulseq)
which in itself is a python adaption of the matlab-based [pulseq](https://github.com/pulseq/pulseq). The documentation
of the open file format for MR sequences can be found [here](https://pulseq.github.io/specification.pdf).
For general usage, please refer to the [pypulseq](https://github.com/imr-framework/pypulseq) documentation.
## Definitions
Please define the offsets and whether you run an m0 scan as follows:
````python
seq.set_definition('offsets_ppm', list(offsets_ppm))
seq.set_definition('run_m0_scan', str(run_m0_scan))
````