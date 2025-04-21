# `AMAZE`: Astrophysical Model Analysis & Evidence Evaluation
A codebase for performing multi-channel inference using catalogs of compact binaries

### Authors:
Michael Zevin, Storm Colloms, April Cheng, Chris Pankow
  
  
### Papers:

https://ui.adsabs.harvard.edu/abs/2017ApJ...846...82Z/abstract

https://ui.adsabs.harvard.edu/abs/2020arXiv201110057Z/abstract

Colloms et al. 2025

Why use one channel when you can use them all? `AMAZE` performs hierarchical inference on branching fractions between any number of population models, where each channel can also be parameterized by physical prescriptions. The executable `model_select` performs the inference, and has many options for including different channels, specifying whether to use mock observations or actual gravitational-wave observations, specifying the prescription for measurement uncertainty, etc. Run `python model_select --help` to learn more about all these options. 

`AMAZE` can now train normalising flows to emulate the input population models, and use the normalising flows for discrete or continuous inference over the model hyperparameters. This currently allows for the population models from Zevin et al. 2020 as input; these models are available on Zenodo here: https://zenodo.org/record/4277620#.X7w28RNKjUI.

These additions were used to produce results in Colloms et al. 2025. The corresponding data release, including the normalising flow models, processed GW public data, and inference output, are available on Zenodo: https://zenodo.org/records/14967688). Future developments will increase the usability of `AMAZE` with normalising flows.

Included in this codebase are a number of notebooks (in the `notebooks/` directory) that were used in pre-processing the data, and generate the figures and numbers from Zevin et al. 2020 (https://ui.adsabs.harvard.edu/abs/2020arXiv201110057Z/abstract). The directory `/plot/flow_results` contains the scripts used to generate the plotting scripts in Colloms et al. 2025.


