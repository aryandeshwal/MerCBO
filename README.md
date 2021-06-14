## Mercer Features for Efficient Combinatorial Bayesian Optimization


This repository contains the source code for the paper "[Mercer Features for Efficient Combinatorial Bayesian Optimization](https://arxiv.org/abs/2012.07762)" published at [AAAI'21](https://aaai.org/Conferences/AAAI-21/) conference. 

In this paper, we propose an efficient approach referred as **Mercer Features for Combinatorial Bayesian Optimization (MerCBO)**. The key idea behind MerCBO is to provide explicit feature maps for diffusion kernels over discrete objects by exploiting the structure of their combinatorial graph representation. These Mercer features combined with Thompson sampling as the acquisition function allows us to employ tractable solvers for finding the next structure for evaluation.



### Requirements/Installation
The code is implemented in Python and requires the [graph-tool](https://graph-tool.skewed.de/) library (other than the ones mentioned in requirements.txt)


### Benchmarks
python main.py --objective labs --n_eval 250


The repository builds upon the [source code](https://github.com/QUVA-Lab/COMBO) provided by the COMBO authors. We thank them for their code and have added appropriate licenses. 


### Citation
If you use this code, please consider citing our paper:
```bibtex
@article{Deshwal_Belakaria_Doppa_2021, 
  title={Mercer Features for Efficient Combinatorial Bayesian Optimization}, volume={35}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/16886}, 
  number={8}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Deshwal, Aryan and Belakaria, Syrine and Doppa, Janardhan Rao}, 
  year={2021}, 
  month={May}, 
  pages={7210-7218}}
````
