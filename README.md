# RNN using Python

A minimal implementation of a **Character-Level Vanilla Recurrent Neural Network (RNN)** built entirely with **NumPy**. This project demonstrates how a recurrent neural network learns to model and generate text one character at a time without relying on deep learning frameworks such as PyTorch or TensorFlow.

The implementation is based on the original educational work by Andrej Karpathy and has been updated for compatibility with current versions of Python and NumPy while preserving the original algorithm.

---

## Features

* Pure NumPy implementation
* Character-level language modeling
* Vanilla RNN with `tanh` activation
* One-hot character encoding
* Forward propagation
* Backpropagation Through Time (BPTT)
* Cross-Entropy loss
* Numerically stable Softmax
* Gradient clipping
* AdaGrad optimizer
* Compatible with Python 3.10+
* Compatible with NumPy 2.x

---

## Project Structure

```text
RNN-using-Python/
├── RNN-using-Python.py
├── input.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Vanshgoe/RNN-using-Python.git
cd RNN-using-Python
```

Install the required dependency:

```bash
pip install numpy
```

or

```bash
pip install -r requirements.txt
```

---

## Dataset

The model trains on any plain text file named:

```text
input.txt
```

Example:

```text
Hello world!

This is a simple character-level language model.
It learns to predict the next character.
```

For better text generation, larger datasets such as Tiny Shakespeare, books, source code, or Wikipedia articles can be used.

---

## Running the Model

Start training with:

```bash
python RNN-using-Python.py
```

The model periodically prints generated text along with the current training loss.

Example output:

```text
============================================================
the king was the best of the world and...
============================================================

Iteration 500 | Loss: 42.13
```

As training continues, the generated text gradually becomes more coherent.

---

## Model Architecture

```text
Input Character
       │
       ▼
One-Hot Encoding
       │
       ▼
Input → Hidden (Wxh)
       │
       ▼
Hidden State (tanh)
       │
       ▼
Hidden → Hidden (Whh)
       │
       ▼
Hidden → Output (Why)
       │
       ▼
Softmax
       │
       ▼
Next Character Prediction
```

---

## Hyperparameters

Default configuration:

```python
hidden_size = 100
seq_length = 25
learning_rate = 0.1
max_iters = 10000
```

These values can be modified depending on the dataset and desired training behavior.

---

## Training Pipeline

Each iteration performs the following steps:

1. Read a sequence of characters.
2. Convert characters into one-hot vectors.
3. Compute the forward pass.
4. Calculate Cross-Entropy loss.
5. Perform Backpropagation Through Time (BPTT).
6. Clip gradients to reduce exploding gradients.
7. Update parameters using AdaGrad.
8. Repeat until training completes.

---

## Mathematical Formulation

Hidden state:

```text
hₜ = tanh(Wxh·xₜ + Whh·hₜ₋₁ + bh)
```

Output logits:

```text
yₜ = Why·hₜ + by
```

Probability distribution:

```text
pₜ = softmax(yₜ)
```

Loss:

```text
L = −Σ log(p(correct character))
```

---

## Learning Objectives

This project demonstrates the fundamental concepts behind recurrent neural networks, including:

* Character-level language modeling
* One-hot encoding
* Hidden state propagation
* Forward propagation
* Backpropagation Through Time (BPTT)
* Cross-Entropy loss
* Gradient clipping
* AdaGrad optimization
* Text generation from scratch

---

## Limitations

This implementation is intentionally minimal and educational.

It does not include:

* LSTM cells
* GRU cells
* Transformer architectures
* Attention mechanisms
* GPU acceleration
* Mini-batch training
* Model checkpointing
* Mixed precision training
* Tokenization
* Distributed training

---

## Future Improvements

Possible extensions include:

* Model checkpoint saving and loading
* Temperature-controlled text sampling
* Command-line interface (CLI)
* Learning rate scheduling
* Mini-batch training
* LSTM implementation
* GRU implementation
* PyTorch version with GPU support
* Interactive text generation

---

## References

* Andrej Karpathy — *Minimal Character-Level RNN*
* *The Unreasonable Effectiveness of Recurrent Neural Networks*
* CS231n: Recurrent Neural Networks
* NumPy Documentation

---

## License

This project is distributed under the BSD License and is based on the original implementation by Andrej Karpathy.

Please retain the original license and attribution when redistributing or modifying the source code.

---

## Acknowledgements

Special thanks to Andrej Karpathy for publishing the original minimal character-level RNN implementation, which has become one of the most widely used educational resources for understanding recurrent neural networks from first principles.
