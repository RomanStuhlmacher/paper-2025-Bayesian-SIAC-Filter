# Numerical experiments

This directory contains all source code required to reproduce the numerical experiments presented in the paper. It is developed for Julia v1.11.4.

The numerical experiments can be found in the _notebooks_ subdirectory of the _Code_ directory. 
The notebooks are organized according to the corresponding sections of the paper.

Access the notebooks by cloning this repository and switch to the _notebooks_ subdirectory:

```
git clone [https://github.com/RomanStuhlmacher/paper-2025-Bayesian-SIAC-Filter.git]
cd paper-2025-Bayesian-SIAC-Filter
cd Code/notebooks
```

To be able to run the notebooks you need `jupyter` or an editor that can execute jupyter notebooks. To execute a notebook using `jupyter` you need the `IJulia` package. To install the `IJulia` package start `julia` in the _notebooks_ directory and run:

```julia
]
add IJulia
```

Then you can open the jupyter notebooks through the terminal via:
```
jupyter notebook
```


