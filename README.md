# DMD (Dynamic Mode Decomposition) Autoencoder
Prediction, estimation, and control of dynamical systems remain challenging due to the nonlinear dynamics most systems hold. However, recent advances in leveraging deep learning to identifying coordinate transformations that make strongly nonlinear dynamics approximately linear have enabled analyzing nonlinear systems. I propose a simple approach to find these coordinate transformations. The proposed approach identifies a nonlinear mapping to a space where dynamics are linear using a deep autoencoder. The autoencoder minimizes the loss of the autoencoder reconstruction and dynamic mode decomposition reconstruction of the latent space trajectories. This simple DMD autoencoder is tested on dynamical system time-series data-sets, including the pendulum and fluid flow around a cylinder.

# Introduction
Predictions of nonlinear dynamical systems is a fundamental problem in engineering. Whenever possible, itis desirable to work in a linear framework. Linear dynamical systems have closed-form solutions. Moreover, there are many techniques for analysis, prediction, and control of linear dynamical systems. In order to transition from a nonlinear framework to a linear framework, we leverage deep learning to learn a nonlinear mapping to a space where the trajectories exhibit approximately linear dynamics. The initial research steps were to reproduce the results in "Deep learning for universal linear embeddings of nonlinear dynamics" by Lusch et al [1]. In this process, I rebuilt their code in an upgraded library Tensorflow 2.0. Since the performance of the Koopman autoencoder highly depended on the weight initialization, I developed a simple Dynamic Mode Decomposition (DMD) autoencoder as a pretrain to Koopman autoencoder model.

# Documentation Site 
https://opaliss.github.io/dmd_autoencoder/

# Dependencies
1. [Python >= 3.7](https://www.python.org/downloads/)
1. [numpy >= 1.19.1](https://numpy.org/install/)
2. [tensorflow >= 2.0](https://www.tensorflow.org/install)
3. [matplotlib >= 3.3.1](https://matplotlib.org/users/installing.html)
4. [pydmd >=0.3](https://pypi.org/project/pydmd/)

# References
[1] [Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. Nature Communications, 9(1):4950, 2018.](https://arxiv.org/pdf/1712.09707.pdf)

[2] [J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, and J. Nathan Kutz. On dynamic mode decomposition: theory and applications. J. Comp. Dyn., 1(2):391-421, 2014.](https://arxiv.org/abs/1312.0041)


# License
[MIT]((https://choosealicense.com/licenses/mit/))
