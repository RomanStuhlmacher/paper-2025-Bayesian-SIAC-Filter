# paper-2025-Bayesian-SIAC-Filter
Reproducibility repository for the paper "The Bayesian SIAC filter"

```latex
@article{glaubitz2025generalized,
  title={The Bayesian SIAC filter},
  author={Glaubitz, Jan and Li, Tongtong and Ryan, Jennifer and
          Stuhlmacher, Roman},
  journal={},
  volume={},
  pages={},
  year={2025},
  month={},
  doi={},
  eprint={},
  eprinttype={},
  eprintclass={}
}
```


If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please also cite this repository as

```latex
@misc{glaubitz2025generalizedRepro,
  title={Reproducibility repository for
         "The Bayesian SIAC filter"},
  author={Glaubitz, Jan and Li, Tongtong and Ryan, Jennifer and
          Stuhlmacher, Roman},
  year={2025},
  howpublished={},
  doi={}
}
```

# Abstract

We propose the Bayesian smoothness-increasing accuracy-conserving (SIAC) filter---a hierarchical Bayesian extension of the existing deterministic SIAC filter. 
The SIAC filter is a powerful numerical tool for removing high-frequency noise from data or numerical solutions without degrading accuracy.
However, current SIAC methodology is limited to (i) nodal data (direct, typically noisy function values) and (ii) deterministic point estimates that do not account for uncertainty propagation from input data to the SIAC reconstruction.
The proposed Bayesian SIAC filter overcomes these limitations by (i) supporting general (non-nodal) data models and (ii) enabling rigorous uncertainty quantification (UQ), thereby broadening the applicability of SIAC filtering.
We also develop structure-exploiting algorithms for efficient maximum a posteriori (MAP) estimation and Markov chain Monte Carlo (MCMC) sampling, with a focus on linear data models with additive Gaussian noise.
Computational experiments demonstrate the effectiveness of the Bayesian SIAC filter across several applications, including signal denoising, image deblurring, and post-processing of numerical solutions to hyperbolic conservation laws. 
The results show that the Bayesian approach produces point estimates with accuracy comparable to, and in some cases exceeding, that of the deterministic SIAC filter. 
In addition, it extends naturally to general data models and provides built-in UQ.


# Numerical experiments

The subfolder _Code_ of this repository contains a README.md file with instructions to reproduce the numerical experiments.

The numerical experiments were carried out using Julia v1.11.4.


# Authors

- [Jan Glaubitz](https://www.janglaubitz.com) (Linköping University, Sweden)
- [Tongtong Li](https://tongtong-li.github.io) (University of Maryland, USA)
- [Jennifer Ryan](https://sites.google.com/view/jennifer-k-ryan/home) (KTH Royal Institute of Technology, Sweden)
- [Roman Stuhlmacher](https://www.kth.se/profile/romanst?l=en) (KTH Royal Institute of Technology, Sweden)

# Disclaimer

Everything is provided as is and without warranty. Use at your own risk!

