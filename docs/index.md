# DMD Autoencoder

Prediction, estimation, and control of dynamical systems remains challenging due to the nonlinear dynamics most systems hold. However, recent advances in leveraging deep learning to identify coordinate transformations that make strongly nonlinear dynamics approximately linear have enabled analyzing nonlinear systems. We propose a simple approach to find these coordinate transformations. The proposed approach identifies a nonlinear mapping to a space where dynamics are linear using a deep autoencoder. The autoencoder minimizes the loss of the autoencoder reconstruction and dynamic mode decomposition reconstruction of the latent space trajectories. This simple DMD autoencoder is tested on dynamical system time series datasets, including the pendulum and fluid flow around a cylinder.

The initial research steps were to reproduce the results in "Deep learning for universal linear embeddings of nonlinear dynamics" by Lusch et al [[1]](https://arxiv.org/pdf/1712.09707.pdf). In this process, we rebuilt their code in an upgraded library Tensorflow 2.0. Since the performance of the Koopman autoencoder highly depended on the weight initialization, we developed a simple Dynamic Mode Decomposition (DMD) autoencoder as a pretrain to Koopman autoencoder model. 

This is a collection of Python subroutines and examples that illustrate how to train a Dynamic Mode Decomposition autoencoder. 

## Dependencies
1. [Python >= 3.7](https://www.python.org/downloads/)
1. [numpy >= 1.19.1](https://numpy.org/install/)
2. [tensorflow >= 2.0](https://www.tensorflow.org/install)
3. [matplotlib >= 3.3.1](https://matplotlib.org/users/installing.html)
4. [pydmd >=0.3](https://pypi.org/project/pydmd/)

## References
[1] [Bethany Lusch, J. Nathan Kutz, and Steven L. Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. Nature Communications, 9(1):4950, 2018.](https://arxiv.org/pdf/1712.09707.pdf)

[2] [J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, and J. Nathan Kutz. On dynamic mode decomposition: theory and applications. J. Comp. Dyn., 1(2):391-421, 2014.](https://arxiv.org/abs/1312.0041)


## License
[MIT](https://choosealicense.com/licenses/mit/)


## Important Python subroutines
The most important Python subroutines are:

  - dmd_machine:
 
        dmd_machine/autoencoder_network.py
        dmd_machine/dmd_ae_machine.py
        dmd_machine/loss_function.py

  - data:

        data/Data.py
   
  - driver/runfile:

        train_discrete_dataset_machine.py
        train_pendulum_machine.py 
        train_fluid_flow_machine.py



## About the Authors
Mathematics Department, San Diego State University 

Research project under the supervision of Professor Christopher Curtis (ccurtis@sdsu.edu). 

Research group: 

- Opal Issan- Applied Mathematics undergraduate student (opal.issan@gmail.com)

- Jay Lago- Computational Science PhD student (jaylago@gmail.com)

- Joseph Diaz- Applied Mathematics Undergraduate student (joseph.a.g.diaz@gmail.com)

- Robby Simpson- Applied Mathematics Masters student (robby.c.simpson@gmail.com)


