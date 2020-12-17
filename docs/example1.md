Consider a simple nonlinear discrete spectrum system, described as follows: 

$$\dot{x_{1}} = \mu x_{1}$$

$$\dot{x_{2}} = \lambda (x_{2} - x_{1}^{2})$$

![](images/ex1_data.png)

Analytic Koopman Embedding 
This system has a three-dimensional Koopman invariant space. 

$$ y_{1} = x_{1} $$

$$ y_{2} = x_{2} $$ 

$$ y_{3} = x_{1}^{2}$$

Therefore, the nonlinear system $x$ can be transformed to a new coordinate system where dynamics are linear 
(in the form of [equation (1) and (2)](dmd.md)). 

$$ \dot{y_{1}} = \dot{x_{1}} = \mu x_{1}  = \mu y_{1} $$

$$ \dot{y_{2}} = \dot{x_{2}} = \lambda (x_{2} - x_{1}^{2}) = \lambda (y_{2} - y_{3}) $$

$$ \dot{y_{3}} = \dot{x_{1}^{2}} = 2 x_{1} \dot{x_{1}} = 2 x_{1} \mu x_{1} = 2 \mu x_{1}^{2} $$ 


\begin{equation} \label{eq:1}
 \frac{d}{dt}
 \begin{bmatrix}
        y_{1}\\
        y_{2}\\
        y_{3}
    \end{bmatrix} = 
    \begin{bmatrix}
        \mu & 0 & 0 \\
        0 & \lambda & -\lambda\\
        0 & 0 & 2\mu
    \end{bmatrix}
    \begin{bmatrix}
        y_{1}\\
        y_{2}\\
        y_{3}
    \end{bmatrix}
\end{equation}

In order to compare the Dynmaic Mode Decomposition reconstruction accuracy on the two coordinates $x$ and $y$, we evaluated 

\begin{equation} \label{eq:6}
    \begin{aligned}
    \left\| X'- AX \right\| _{F}^{2} = \left\| X'- (X' V  \Sigma^{-1} U^{*}) X \right\| _{F}^{2}\\
    = \left\| X'- (X'  V  \Sigma^{-1}  U^{*}) ( U \Sigma  V^{T}) \right\| _{F}^{2} \\  =\left\| X' ( I - V V^{T}) \right\| _{F}^{2}
    \end{aligned}
\end{equation}

where in the $y$ coordinate system dynamic mode decomposition reconstruction was able to recover twice as many significant figures in comparison to the $x$ coordinate. 

- Related notebooks: `dmd_autoencoder_discrete_train.ipynb` and `compare_full_machine_results_discrete_dataset.ipynb` 

## DMD Autoencoder Results

The Dynamic Mode Decomposition Autoencoder attempted to find a nonlinear mapping $g$ which is also called the encoder to a space where the dynamics are linear. While the analysis about shows that the Koopman invariant space is three-dimensional for this example, we trained the model to have a latent space of two-dimensions. 

<!---  As a result, $L_{2} =  3.27 × 10^{-4}$ and $L_{3} = 5.055 × 10^{-5}$, whereas in the latent space dataset $Y$, $L_{2} =  3.006 × 10^{-4}$ and $L_{3} = 2.703 × 10^{-4}$. While the DMD loss decreased, the predictability loss did not improve. There is more room for improvement by adjusting the network's hyper-parameters. --->
