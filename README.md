# Non-intrusive surrogate modeling for parametrized time-dependent partial differential equations using convolutional autoencoders

![Slide1](https://user-images.githubusercontent.com/15322711/136744148-96d37d6f-5350-4deb-a219-b21d1c73fa6a.jpg)

This paper presents a novel non-intrusive surrogate modeling scheme based on deep learning for predictive modeling of complex systems, described by parametrized time-dependent partial differential equations. Specifically, the proposed method utilizes a convolutional autoencoder in conjunction with a feed forward neural network to establish a mapping from the problem’s parametric space to its solution space. For this purpose, training data are collected by solving the high-fidelity model via finite elements for a reduced set of parameter values. Then, by applying the convolutional autoencoder, a low-dimensional vector representation of the high dimensional solution matrices is provided by the encoder, while the reconstruction map is obtained by the decoder. Using the latent vectors given by the encoder, a feed forward neural network is efficiently trained to map points from the parametric space to the compressed version of the respective solution matrices. This way, the proposed surrogate model is capable of predicting the entire time history response simultaneously with remarkable computational gains and very high accuracy. The elaborated methodology is demonstrated on the stochastic analysis of time-dependent partial differential equations solved with the Monte Carlo method.

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
