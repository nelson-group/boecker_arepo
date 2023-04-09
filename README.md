Arepo public version
====================

AREPO is a massively parallel code for gravitational n-body
systems and hydrodynamics, both on Newtonian as well as
cosmological background. It is a flexible code that can be
applied to a variety of different types of simulations, offering
a number of sophisticated simulation algorithms.

This version of AREPO includes a supernova feedback model, based on the BSc thesis of Xeno Boecker at Heidelberg University (2023).

The following `Config.sh` options are added:

```
#XENO_SN
#SN_TYPE=0	# 0: flat energy injection; 1: flat momentum injection; 2: scaled energy injection; 3: scaled momentum injection
#FLAT_ENERGY_SN
#FLAT_MOMENTUM_SN
#SCALED_ENERGY_SN
#SCALED_MOMENTUM_SN
```

See analysis and visualization routines in the `jupyterNotebooks/` directory.
