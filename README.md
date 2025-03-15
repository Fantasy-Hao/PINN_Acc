# Accelerated Training Methods of Physics-Informed Neural Networks(PINNs)


The repository contains the implementations of the graduation design(thesis) entitled “Accelerated Training Methods of Physics-Informed Neural Networks” by Yuxin Hao. 

Physics-Informed Neural Networks (PINN) is a machine learning framework that combines deep learning with prior physical knowledge to solve complex physics problems, particularly the solution of Partial Differential Equations (PDE) and the simulation of system dynamics. However, in practical scenarios, the training process of the PINN model often encounters challenges like sluggish convergence and suboptimal computational efficiency.

This thesis provides a comprehensive introduction to the fundamental theory of PINN and its applications in handling complex PDE, contrasts PINN with traditional machine learning methods, and analyzes the advantages of PINN over traditional numerical methods. Subsequently, with a focus on the phenomenon of vortex shedding as delineated by the Navier-Stokes equation, specific solutions are proposed, and optimizations for PINN are explored along three avenues: adaptive sampling, dynamic adjustment of loss weights, and enhanced network architecture.
    
In terms of sampling methods, the thesis initially introduces traditional fixed sampling and random resampling, followed by two types of adaptive sampling strategies: adaptive distribution based on residuals effectively enhances model prediction accuracy by simulating regions with large residuals, while the retain-resample-release method further reduces computational burden. For dynamic adjustment of loss weights, the adaptability of neural networks and residual-based attention are incorporated into the loss function, accelerating the model convergence process. In terms of network structure, an improved Multi-Layer Perceptron based on Transformers enhances the model's ability to capture complex data features by combining residual connections and self-attention mechanisms. Through the integration of these three methods, optimization and acceleration of the PINN model are achieved.
    
Finally, the thesis summarizes the principal research findings and proposes several future directions for further enhancing the PINN algorithm.


## Result
<figure>
    <figcaption>MSE Curve</figcaption>
    <img src="./result/loss_curve/ablation_loss.png" width="100%" height="100%"/>
</figure>

<table>
    <tr>
        <td>
            <figure>
                <figcaption>Exact solution: u </figcaption>
                <img src="./result/solutions/u_star.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Fitting solution: u </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_u_pred.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Absolute error: u </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_u_error.png" width="100%" height="100%"/>
            </figure>
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <figcaption>Exact solution: v </figcaption>
                <img src="./result/solutions/v_star.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Fitting solution: v </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_v_pred.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Absolute error: v </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_v_error.png" width="100%" height="100%"/>
            </figure>
        </td>
    </tr>
    <tr>
        <td>
            <figure>
                <figcaption>Exact solution: p </figcaption>
                <img src="./result/solutions/p_star.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Fitting solution: p </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_p_pred.png" width="100%" height="100%"/>
            </figure>
        </td>
        <td>
            <figure>
                <figcaption>Absolute error: p </figcaption>
                <img src="./result/solutions/sa+rba+mmlp_p_error.png" width="100%" height="100%"/>
            </figure>
        </td>
    </tr>
</table>


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/gh-pages/MIT-LICENSE.txt)


## Reference
 - Raissi M, Perdikaris P, Karniadakis G E. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J].](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) Journal of Computational physics, 2019, 378: 686-707.
 - Wu C, Zhu M, Tan Q, et al. [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0045782522006260) Computer Methods in Applied Mechanics and Engineering, 2023, 403: 115671.
 - Lu L, Meng X, Mao Z, et al. [DeepXDE: A deep learning library for solving differential equations[J].](https://epubs.siam.org/doi/abs/10.1137/19M1274067) SIAM review, 2021, 63(1): 208-228.
 - Daw A, Bu J, Wang S, et al. [Mitigating propagation failures in physics-informed neural networks using retain-resample-release (r3) sampling[J].](https://arxiv.org/abs/2207.02338) arXiv preprint arXiv:2207.02338, 2022.
 - McClenny L D, Braga-Neto U M. [Self-adaptive physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0021999122007859) Journal of Computational Physics, 2023, 474: 111722.
 - Anagnostopoulos S J, Toscano J D, Stergiopulos N, et al. [Residual-based attention in physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000616) Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116805.
 - Wang S, Teng Y, Perdikaris P. [Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J].](https://epubs.siam.org/doi/abs/10.1137/20M1318043) SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081.
