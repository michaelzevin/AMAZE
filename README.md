# `AMAZE`: Astrophysical Model Analysis & Evidence Evaluation
A codebase for performing multi-channel inference using catalogs of compact binaries

### Authors:
Michael Zevin, Storm Colloms, April Cheng, Chris Pankow
  
  
### Papers:

https://ui.adsabs.harvard.edu/abs/2017ApJ...846...82Z/abstract

https://ui.adsabs.harvard.edu/abs/2020arXiv201110057Z/abstract

https://ui.adsabs.harvard.edu/abs/2025arXiv250303819C/abstract

Why use one channel when you can use them all? `AMAZE` performs hierarchical inference on branching fractions between any number of population models, where each channel can also be parameterized by physical prescriptions. The executable `amaze` performs the inference, and has many options for including different channels, specifying whether to use mock observations or actual gravitational-wave observations, specifying the prescription for measurement uncertainty, etc. See `examples/AMAZE-example.ini` to learn more about all these configuration options. 

`AMAZE` can now train normalising flows to emulate the input population models, and use the normalising flows for discrete or continuous inference over the model hyperparameters. This allows for any population synthesis models as input, with formation channels, hyperparameters, and event parameters specified as shown in `examples/AMAZE-example.ini`.

These additions were used to produce results in Colloms et al. 2025, using population synthesis models from Zevin et al. 2020 as input (available on Zenodo here: https://doi.org/10.5281/zenodo.4277619). The corresponding data release, including the configuration files necessary to reproduce paper results, the normalising flow models, processed GW public data, and inference output, are available on Zenodo: https://doi.org/10.5281/zenodo.14967687.
