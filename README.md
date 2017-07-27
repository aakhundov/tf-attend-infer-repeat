Attend, Infer, Repeat
=====================

Implementation of **AIR** framework proposed in ["Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"](https://arxiv.org/abs/1603.08575) paper by Eslami et al (DeepMind). The implementation is still work in progress. Gumbel-Softmax (Concrete) distribution proposed [here](https://arxiv.org/abs/1611.01144) and [here](https://arxiv.org/abs/1611.00712) is used to backpropagate throguh discrete random sampling of "z_pres" instead of the [NVIL](https://arxiv.org/abs/1402.0030) used in the original paper.

* **multi_mnist.py** needs to be run first for generation of multi-MNIST dataset.
* **model.py** is a runnable script containing model definition and training code in TensorFlow.
* **transformer.py** borrowed from [TensorFlow models repository](https://github.com/tensorflow/models/tree/master/transformer).
* **gumbel.py** borrowed from [Eric Jang's Gist](https://gist.github.com/ericjang/1001afd374c2c3b7752545ce6d9ed349).
