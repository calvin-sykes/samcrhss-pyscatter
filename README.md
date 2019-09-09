# samcrhss-pyscatter

This is a python implementation of a MCRT uniform sphere scattering code, including the forced first scattering and peeling-off alogrithms for variance reduction. 

To run it you'll need Python 3, and the libraries numpy, numba and matplotlib installed. There's a Jupyter notebook and a standalone Python script version. Both versions will run a series of calculations for different optical depths, with and without the variance reduction enabled. The notebook will plot the resulting images inline, while the standalone script will save a couple of PDFs, examples of which are also included in this repository. In case you don't have numba, I've included numpy-only versions as well, which will be 20-25% slower but are otherwise identical.

The code uses a few tricks to be reasonably fast, and it should only take a minute or two to run. Especially with peeling-off turned on, the slowest part is generating the plot!
