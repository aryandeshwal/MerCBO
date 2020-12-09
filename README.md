## Mercer Features for Efficient Combinatorial Bayesian Optimization


This repository contains the source code for the paper "[Mercer Features for Efficient Combinatorial Bayesian Optimization]()" which is accepted for publication at [AAAI'21](https://aaai.org/Conferences/AAAI-21/). In this paper, we propose an efficient approach referred as **Mercer Features for Combinatorial Bayesian Optimization (MerCBO)**. 
The key idea behind MerCBO is to provide explicit feature maps for diffusion kernels over discrete objects by exploiting the structure of their combinatorial graph representation. These Mercer features combined with Thompson sampling as the acquisition function allows us to employ tractable solvers for finding the next structure for evaluation.
The repository builds upon the [source code](https://github.com/QUVA-Lab/COMBO) provided by the COMBO authors. We thank them for their code and have added appropriate licenses. 



### Requirements/Installation
The code is implemented in Python and requires the following key library other than the ones mentioned in requirements.txt:[graph-tool](https://graph-tool.skewed.de/) 


### Benchmarks
python main.py --objective labs --n_eval 250
