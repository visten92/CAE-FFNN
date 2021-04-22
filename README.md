# Non-intrusive Surrogate Modeling for Parametrized Time-Dependent PDEs using Convolutional Autoencoders

This work presents  a non-intrusive surrogate modeling scheme based on machine learning technology for predictive modeling of complex systems, described by parametrized time-dependent PDEs.  For this type ofproblems, typical finite element solution approaches involve the spatiotemporal discretization of the PDE and the solution of the corresponding linear system of equations at each time step.  Instead, the proposedmethod utilizes a Convolutional Autoencoder in conjunction with a feed forward Neural Network to establish a  low-cost  and  accurate  mapping  from  the  problem’s  parametric  space  to  its  solution  space.   The  aim  isto evaluate directly the entire time history solution matrix through these interpolation schemes.  For this purpose, time history response data are collected by solving the high-fidelity model via FEM for a reduced setof parameter values.  Then, by applying the Convolutional Autoencoder to this data set, a low-dimensional representation of the high-dimensional solution matrices is provided by the encoder, while the reconstruction map is obtained by the decoder.  Using the latent representation given by the encoder, a feed forward Neural Network is efficiently trained to map points from the problem’s parametric space to the compressed version of  the  respective  solution  matrices.   This  way,  the  encoded  time-history  response  of  the  system  at  new parameter  values  is  given  by  the  Neural  Network,  while  the  entire  high-dimensional  system’s  response  is delivered by the decoder.  This approach effectively bypasses the need to serially formulate and solve the governing equations of the system at each time increment, thus resulting in a significant computational costreduction and rendering the method ideal for problems requiring repeated model evaluations or ’real-time’computations.  The elaborated methodology is demonstrated on the stochastic analysis of time-dependent PDEs solved with the Monte Carlo method, however, it can be straightforwardly applied to other similar-type problems, such as sensitivity analysis, design optimization, etc.

* Stefanos Nikolopoulos, Ioannis Kalogeris, Vissarion Papadopoulos [Non-intrusive surrogate modeling for parametrized time-dependent PDEs using Convolutional Autoencoders](https://arxiv.org/abs/2101.05555) arXiv preprint arXiv:2101.05555 (2021)

## Citation

      @misc{nikolopoulos2021nonintrusive,
      title={Non-intrusive surrogate modeling for parametrized time-dependent PDEs using convolutional autoencoders}, 
      author={Stefanos Nikolopoulos and Ioannis Kalogeris and Vissarion Papadopoulos},
      year={2021},
      eprint={2101.05555},
      archivePrefix={arXiv},
      primaryClass={math.NA}
      }
