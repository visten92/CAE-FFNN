# Non-intrusive surrogate modeling for parametrized time-dependent partial differential equations using Convolutional Autoencoders

![Slide1](https://user-images.githubusercontent.com/15322711/136744148-96d37d6f-5350-4deb-a219-b21d1c73fa6a.jpg)

This work presents  a non-intrusive surrogate modeling scheme based on Machine Learning technology for predictive modeling of complex systems, described by parametrized time-dependent PDEs.  For this type of problems, typical finite element solution approaches involve the spatiotemporal discretization of the PDE and the solution of the corresponding linear system of equations at each time step.  Instead, the proposed method utilizes a Convolutional Autoencoder in conjunction with a feed forward Neural Network to establish a  low-cost  and  accurate  mapping  from  the  problem’s  parametric  space  to  its  solution  space.   The  aim  is to evaluate directly the entire time history solution matrix through these interpolation schemes.  For this purpose, time history response data are collected by solving the high-fidelity model via FEM for a reduced set of parameter values.  Then, by applying the Convolutional Autoencoder to this data set, a low-dimensional representation of the high-dimensional solution matrices is provided by the encoder, while the reconstruction map is obtained by the decoder.  Using the latent representation given by the encoder, a feed forward Neural Network is efficiently trained to map points from the problem’s parametric space to the compressed version of  the  respective  solution  matrices.   This  way,  the  encoded  time-history  response  of  the  system  at  new parameter  values  is  given  by  the  Neural  Network,  while  the  entire  high-dimensional  system’s  response  is delivered by the decoder.  This approach effectively bypasses the need to serially formulate and solve the governing equations of the system at each time increment, thus resulting in a significant computational cost reduction and rendering the method ideal for problems requiring repeated model evaluations or ’real-time’ computations.  The elaborated methodology is demonstrated on the stochastic analysis of time-dependent PDEs solved with the Monte Carlo method, however, it can be straightforwardly applied to other similar-type problems, such as sensitivity analysis, design optimization, etc.

* Stefanos Nikolopoulos, Ioannis Kalogeris, Vissarion Papadopoulos ["Non-intrusive surrogate modeling for parametrized time-dependent partial differential equations using convolutional autoencoders"](https://www.sciencedirect.com/science/article/abs/pii/S0952197621004541?via%3Dihub) 

* Link to download the datasets: https://drive.google.com/drive/folders/1uHODdUYDosjCSPZ3EuapqNtJLNIbS__W

## Citation

      @article{NIKOLOPOULOS2022104652,
      title = {Non-intrusive surrogate modeling for parametrized time-dependent partial differential equations using convolutional autoencoders},
      journal = {Engineering Applications of Artificial Intelligence},
      volume = {109},
      pages = {104652},
      year = {2022},
      issn = {0952-1976},
      doi = {https://doi.org/10.1016/j.engappai.2021.104652},
      url = {https://www.sciencedirect.com/science/article/pii/S0952197621004541},
      author = {Stefanos Nikolopoulos and Ioannis Kalogeris and Vissarion Papadopoulos}
      }
