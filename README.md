# conformal_lightgbm_tweedie
Conformal Interval Estimation in Regularized Insurance Models

## Citation

If you use this repository, please cite our work:

```bibtex
@article{https://doi.org/10.1002/asmb.70045,
author = {Manna, Alokesh and Vikram Sett, Aditya and Dey, Dipak K. and Gu, Yuwen and Schifano, Elizabeth D. and He, Jichao},
title = {Conformal Prediction Inference in Regularized Insurance Models},
journal = {Applied Stochastic Models in Business and Industry},
volume = {41},
number = {5},
pages = {e70045},
keywords = {conformal inference, Generalized Linear Model, LightGBM, selective inference, Tweedie regression, Uncertainty Quantification},
doi = {https://doi.org/10.1002/asmb.70045},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/asmb.70045},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/asmb.70045},
abstract = {ABSTRACT Prediction uncertainty quantification has become a key research topic in recent years, with applications in both scientific and business problems. In the insurance industry, assessing the range of possible claim costs for individual drivers improves premium pricing accuracy. It also enables insurers to manage risk more effectively by accounting for uncertainty in accident likelihood and severity. In the presence of covariates, a variety of regression-type models are often used for modeling insurance claims, ranging from relatively simple generalized linear models (GLMs) to regularized GLMs to gradient boosting models (GBMs). Conformal predictive inference has arisen as a popular distribution-free approach for quantifying predictive uncertainty under relatively weak assumptions of exchangeability, and has been well studied under the classic linear regression setting. In this work, we leverage GLMs and GBMs to define meaningful non-conformity measures, which are then used within the conformal prediction framework to provide reliable uncertainty quantification for these types of regression problems. Using regularized Tweedie GLM regression and LightGBM with Tweedie loss, we demonstrate conformal prediction performance with these non-conformity measures in insurance claims data. Our simulation results favor the use of locally weighted Pearson residuals for LightGBM over other methods considered, as the resulting intervals maintained the nominal coverage with the smallest average width.},
year = {2025}
}



```bibtex
@article{manna2025distributionfreeinferencelightgbmglm,
  title={Distribution-free inference for LightGBM and GLM with Tweedie loss},
  author={Alokesh Manna and Aditya Vikram Sett and Dipak K. Dey and Yuwen Gu and Elizabeth D. Schifano and Jichao He},
  year={2025},
  eprint={2507.06921},
  archivePrefix={arXiv},
  journal={arXiv preprint arXiv:2507.06921},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2507.06921},
}
