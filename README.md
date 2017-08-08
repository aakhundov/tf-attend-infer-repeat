Attend, Infer, Repeat
=====================

Implementation of **AIR** framework proposed in ["Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"](https://arxiv.org/abs/1603.08575) paper by Eslami et al (DeepMind). The implementation is still work in progress. Gumbel-Softmax (Concrete) distribution proposed [here](https://arxiv.org/abs/1611.01144) and [here](https://arxiv.org/abs/1611.00712) is used to back-propagate through discrete random sampling of "z_pres" instead of the [NVIL](https://arxiv.org/abs/1402.0030) used in the original paper. Implemented in TensorFlow.

* **multi_mnist.py** needs to be run before training the model for generation of multi-MNIST dataset.
* **main.py** is a runnable script to run the model training code.
* **air_model.py** contains widely configurable AIRModel class.
* **transformer.py** borrowed from [TensorFlow models repository](https://github.com/tensorflow/models/tree/master/transformer).
