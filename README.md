### Adaptive SIR models 
:bug:

We study the adaptive human behavior model implemented in the article [Adaptive human behavior in epidemiological models](https://www.pnas.org/doi/full/10.1073/pnas.1011250108#:~:text=Adaptive%20behavior%20implies%20disease%20transmission,once%20a%20disease%20has%20emerged.)

- `notes/` contains master's thesis proposal.
- `references/` contains main related references.
- `code/` contains disaggregated contact rates and adaptive algorithms.

    - `code/disaggregated`: contains code and simulations for article: [A nonlinear relapse model with disaggregated contact rates: analysis of a forward-backward bifurcation](https://arxiv.org/abs/2302.00161)
    - `code/adaptive`: contains an implementation of adaptive algorithms for epidemiological models with relapse. The module `MDP` implements a simple finite horizon *Markov Decision Process*, and the module `adaptive_MDP` uses this to solve epidemiological models.
    - In each section please review `source_code` for the code modules and `experiments` for some jupyter notebooks and scripts computing images for research and simulations.

JCM