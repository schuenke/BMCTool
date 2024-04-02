======================
Guide for Contributors
======================

Repo structure
==============
This repository uses a *pyproject.toml* file to specify all the requirements.

**.github/workflows**
    Definitions of GitHub action workflows to carry out formatting checks, run tests and automatically create this
    documentation.

**.docs**
    Files to create this documentation.

**src/bmctool**
    Main code for this package

**tests**
    Tests which are automatically run by pytest.
    The subfolder structure should follow the same structure as in *src/bmctool*.


src/bmctool structure
===================
**library**
    Some example sequence and config files used in the examples/tests.

**parameters**
    All (data) classes used to store parameters for the different magnetization pools, simulation options, etc.

**simulation**
    All code to perform the actual simulations by solving the Bloch-McConnell equations.

**utils**
    Utilities such as special rf pulses, evaluation functions, etc.
