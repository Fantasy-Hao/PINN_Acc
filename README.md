# PINN_Acc

> **Accelerated Training Methods for Physics-Informed Neural Networks (PINNs)**

## Overview

Physics-Informed Neural Networks (PINNs) incorporate physical laws—typically expressed by partial differential equations (PDEs)—into neural network training. By embedding PDE residuals into the loss function, PINNs can solve forward or inverse problems even with limited or no observational data.

However, vanilla PINNs often suffer from **slow convergence, training instability, and poor efficiency**, especially when solving complex PDEs such as the Navier–Stokes equations.

This repository implements multiple **training-acceleration techniques** proposed in the related research/thesis, aiming to substantially improve the efficiency and accuracy of PINN-based PDE solvers.

---


## Methods

This project focuses on enhancing PINNs for PDE solving—particularly for the Navier–Stokes equations—through improvements in sampling, loss design, and neural architectures.

### **1. Adaptive Sampling Methods**

- **RAR (Residual-based Adaptive Refinement)**  
  Dynamically adds collocation points in high-residual regions.

- **RAD (Residual-based Adaptive Distribution)**  
  Redistributes sampling points based on residual probability distribution.

- **R3 (Retain–Resample–Release)**  
  Addresses the inefficiency and high sensitivity of RAD by iteratively:
  - Retaining high-error points  
  - Resampling uncertain regions  
  - Releasing outdated points  
  → Achieves faster convergence and improved stability.

### **2. Improved Loss Functions**

- **Dynamic Weighting**  
  Automatically balances PDE residual loss and data loss during training.

- **RBA (Residual-Based Attention)**  
  Assigns attention weights according to residual variations *without extra gradient computation*, improving robustness and accuracy.

### **3. Network Architecture Enhancements**

- **mMLP (Modified MLP)**  
  A refined feed-forward architecture optimized for PDE learning.

- **Self-Attention Integration**  
  Enhances representation capability for complex spatial interactions.

---


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/git/git-scm.com/blob/gh-pages/MIT-LICENSE.txt)

---


## References
 - Raissi M, Perdikaris P, Karniadakis G E. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J].](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125) Journal of Computational physics, 2019, 378: 686-707.
 - Wu C, Zhu M, Tan Q, et al. [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0045782522006260) Computer Methods in Applied Mechanics and Engineering, 2023, 403: 115671.
 - Lu L, Meng X, Mao Z, et al. [DeepXDE: A deep learning library for solving differential equations[J].](https://epubs.siam.org/doi/abs/10.1137/19M1274067) SIAM review, 2021, 63(1): 208-228.
 - Daw A, Bu J, Wang S, et al. [Mitigating propagation failures in physics-informed neural networks using retain-resample-release (r3) sampling[J].](https://arxiv.org/abs/2207.02338) arXiv preprint arXiv:2207.02338, 2022.
 - McClenny L D, Braga-Neto U M. [Self-adaptive physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0021999122007859) Journal of Computational Physics, 2023, 474: 111722.
 - Anagnostopoulos S J, Toscano J D, Stergiopulos N, et al. [Residual-based attention in physics-informed neural networks[J].](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000616) Computer Methods in Applied Mechanics and Engineering, 2024, 421: 116805.
 - Wang S, Teng Y, Perdikaris P. [Understanding and mitigating gradient flow pathologies in physics-informed neural networks[J].](https://epubs.siam.org/doi/abs/10.1137/20M1318043) SIAM Journal on Scientific Computing, 2021, 43(5): A3055-A3081.
