# Probabilistic Perspective Machine Learning ppml [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository contains code to replicate, modify the codes and prove the mathematical concepts from 
1. [probml/pmtk3](https://github.com/probml/pmtk3)
2. [Machine Learning A Probabilistic Perspective](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf).  


All the work is in the following format, 
- MATLAB implementation
- Python implementation with proofs and explanations

This repository is intended for people who want to learn more about Probabilistic Machine Learning alongside with simplified examples, and will cover different concepts, such as [Number Game Bayes](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/3GMDD/F3.2/3.2numberGame.ipynb), [Monte Carlo Sampling Pi](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/2Probability/F2.19/2.19mcEstimatePi.ipynb), [naive Bayes classifier](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/3GMDD/F3.8/3.8naiveBayesBowDemo.ipynb), [student distribution em algorithm](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/2Probability/F2.8/2.8RobustDemo.ipynb), [Principal Component Analysis](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/12LatentLinearModels/F12.5/12.5pcaImageDemo.ipynb), [Independent Component Analysis](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/12LatentLinearModels/F12.20/12.20icaDemo.ipynb), GMM EM, lasso, bayes net, etc.

Other relevant works include, [Local Interpretable Model-Agnostic Explanations](https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/LIME/LIME.ipynb), Bata-VAE, Neural Statistician, Gaussian Dropout, Variational Dropout, Bayes by Backprop, InfoGAN, etc.

 ## References
 - Murphy, Kevin P. 2012, Machine Learning: A Probabilistic Perspective, The MIT Press 0262018020, 9780262018029. 
 - Ribeiro, Marco Tulio et al. “"Why Should I Trust You?": Explaining the Predictions of Any Classifier.” HLT-NAACL Demos (2016).
 - Burgess, Christopher P. et al. “Understanding disentangling in $\beta$-VAE.” (2018).
 - Kingma, Diederik P. and Max Welling. “Auto-Encoding Variational Bayes.” CoRR abs/1312.6114 (2013): n. pag.
 
 ## Results
LIME         |  ICA
:-------------------------:|:-------------------------:
<img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/LIME/result.png" width="320" > |  <img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/12LatentLinearModels/F12.20/icaresult.png" width="320" >
PCA|MC Pi estimation
<img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/12LatentLinearModels/F12.5/pcaresult.png" width="320" >|<img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Machine%20Learning%20A%20Probabilistic%20Perspective/2Probability/F2.19/result.png" width="320" >
Gaussian Blob dataset|VAE
<img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Variational%20Autoencoder%20and%20Its%20extension/BVAE/GaussBlob.gif" width="320" >|<img src="https://github.com/zcemycl/ProbabilisticPerspectiveMachineLearning/blob/master/Variational%20Autoencoder%20and%20Its%20extension/VAE/VAEmnist.gif" width="320" >

