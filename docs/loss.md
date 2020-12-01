The simple DMD autoencoder loss function is a combination of three evaluations: 

**Autoencoder reconstruction loss** -  This ensures that the original trajectories can be recovered.
\begin{equation} \label{eq:8}
L_{1} = MSE \left\| X - g^{-1}(\tilde{Y}) \right\|
\end{equation}

**DMD loss** - evaluate the linearity of the latent space dynamics. The following is derived in equation (6).

\begin{equation} \label{eq:9}
L_{2} = \left\| Y' ( I - V V^{T}) \right\| _{F}^{2}
\end{equation}


**DMD reconstruction loss** - evaluate the DMD least squares fit of the $A$ matrix. 
\begin{equation} \label{eq:10}
L_{3} = MSE\left\| Y - \tilde{Y} \right\| 
\end{equation}

The final loss function is $L = \alpha_{1}L_{1} + \alpha_{2}L_{2} + \alpha_{3}L_{3}$. 
