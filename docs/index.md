# Enhancing Dynamic Mode Decomposition using Autoencoder Networks

## Abstract 
Prediction, estimation, and control of dynamical systems remains challenging due to nonlinearity. The Koopman operator is an infinite-dimensional linear operator that evolves the observables of a dynamical system which we approximate by the dynamic mode decomposition (DMD) algorithm. Using DMD to predict the evolution of a nonlinear dynamical system over extended time horizons requires choosing the right observable function defined on the state space. A number of DMD modifications have been developed to choose the right observable function, such as Extended DMD.  Here, we propose a simple machine learning based approach to find these coordinate transformations.
This is done via a deep autoencoder network. This simple DMD autoencoder is tested and verified on nonlinear dynamical system time series datasets, including the pendulum and fluid flow past a cylinder.

*Keywords* - Dynamic mode decomposition, Deep learning, Dynamical systems, Koopman analysis, Observable functions.

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
This is a collection of Python subroutines and examples that illustrate how to train a Dynamic Mode Decomposition autoencoder. 

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


