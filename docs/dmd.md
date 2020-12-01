		
# Dynamic Mode Decomposition and the Koopman Operator
     
Let $x_{t}$ be the state vector of a nonlinear dynamic system. In order to create a linear model, our goal is to fit the dynamical system states to a model of the form:

(1) $$\frac{d}{d t} x = Ax $$

(2) $$x_{t+1} = Ax_{t} $$
    
A nonlinear system can be represented in term of an infinite dimensional operator acting on a Hilbert space of measurement function of the state of the system. The Koopman operator is linear, yet infinite-dimensional. An approximation of the Koopman Operator can be obtained by variants of the Dynamic Mode Decomposition algorithm. 

The Dynamic Mode Decomposition developed by Schmid is a dimensionality reduction algorithm. Given time series dataset, the exact Dynamic Mode Decomposition computes the best fit operator A that advances the system measurements in time [[2]](https://arxiv.org/abs/1312.0041).
The time series dataset can be arranged into two matrices, $X$ and $X'$:
    
(3)

\begin{equation} \label{eq:3}
 X=
    \begin{bmatrix}
        \vert & \vert &  &  \vert\\
        x_{0}   & x_{1} & ...& x_{m-1} \\
        \vert & \vert &  & \vert 
    \end{bmatrix}
\end{equation}


(4) 

\begin{equation} \label{eq:4}
 X' = 
    \begin{bmatrix}
        \vert & \vert &  &  \vert\\
        x_{1}   & x_{2} & ...& x_{m} \\
        \vert & \vert &  & \vert 
    \end{bmatrix}
\end{equation}

    
In order to find the matrix A in equation (1) and (2), we solve the linear system with the DMD algorithm. By the singular value decomposition, $X \approx U \Sigma V^{*}$ where $\tilde{U} \in \mathbb{C}^{n \times r}$, $\tilde{\Sigma} \in \mathbb{C}^{r \times r}$, and $\tilde{V} \in \mathbb{C}^{m \times r}$. Therefore, the matrix A is obtained by $A = X'\tilde{V}\tilde{\Sigma}^{-1}\tilde{U}^{*}$. 

The DMD objective is to minimize the following:

(5) 

\begin{equation} \label{eq:5}
\arg \min _{A}\left\| X' - A X \right\| _{F}^{2}
\end{equation}

There are many ways to measure the accuracy of the DMD fit, a simple way is to evaluate the following expression:

(6) 

\begin{equation} \label{eq:6}
    \begin{aligned}
    \left\| X'- AX \right\| _{F}^{2} = \left\| X'- (X' V  \Sigma^{-1} U^{*}) X \right\| _{F}^{2}\\
    = \left\| X'- (X'  V  \Sigma^{-1}  U^{*}) ( U \Sigma  V^{T}) \right\| _{F}^{2} \\  =\left\| X' ( I - V V^{T}) \right\| _{F}^{2}
    \end{aligned}
\end{equation}
    
An additional approach to evaluate the DMD fit is by comparing the DMD reconstruction data to the time-series input data $X$. The DMD reconstruction can be obtained in two different approaches. The first approach is by taking powers of the matrix $A$. A more efficient approach is by expanding the system state in terms of the data-driven spectral decomposition. 

(7) \begin{equation} \label{eq:7}
    x_{k} = \sum_{i=1}^{r} \phi_{i} \lambda_{i}^{k-1} b_{i} = \Phi \Lambda^{k-1} b
\end{equation}

Where $\Phi$ are the eigenvectors of the $A$ matrix, $\lambda$ are the eigenvalues of the $A$ matrix, and b is the mode amplitude. Hence, the DMD reconstruction loss can be computed by the mean squared error of the difference between the input data $X$ and the spectral decomposition in equation (7). 