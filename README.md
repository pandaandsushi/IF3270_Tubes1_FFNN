# IF3270_Tubes1_FFNN - Feedforward Neural Network
A simple FFNN algorithm made from scratch in python

## Brief Overview 
In the development of artificial intelligence models, especially neural networks, Feedforward Neural Network (FFNN) is one of the most widely used architectures due to its simplicity and ability to handle various classification and regression tasks. FFNN works by flowing data unidirectionally from the input layer to the output layer through one or more hidden layers, where each neuron in the network has weights and biases that are updated during the training process.

This implementation provides a flexible and customizable FFNN model, where you can define the architecture (such as depth, width, etc), activation functions, loss functions, and weight & biases initialization methods according to your specific needs. Unlike ready-made frameworks, this model is built **from scratch**, using mathematical computations handled manually through modules like Numpy.

However, it’s important to note that while this implementation meets the essential requirements of a functional FFNN model, it is not yet fully optimized. There are areas for improvement, such as enhancing the efficiency of the computations, adding more sophisticated techniques for weight & biases initialization and activation, and optimizing the training process. Room for improvement remains, but the current version still provides a solid foundation for learning about neural networks and their underlying mechanics.

## Features

### Customizable Neural Network Architecture:
- The model basicly accepts a list of integers that represent the number of neurons in each layer (input, hidden, and output).
- Example:

    ```python
    # act_function: sigmoid; depth: 3; width: 100
    model = FFNN(layers=[784, 100, 100, 10], activation_functions=['sigmoid', 'sigmoid', 'sigmoid'])
    ```

### Activation Functions:
- Fully customizable activation functions for each layer, including:
  - **Linear**
  - **ReLU**
  - **Sigmoid**
  - **Hyperbolic Tangent (tanh)**
  - **Softmax**
  
- Additional activation functions:
  - **ELU (Exponential Linear Unit)**
  - **Swish**
  - **SoftPlus**

### Loss Functions:

- Fully customizable loss functions, including:
  - **Mean Squared Error (MSE)**
  - **Binary Cross-Entropy**
  - **Categorical Cross-Entropy**

### Weight Initialization:
- Fully customizable weight initialization methods:
  - **Zero Initialization**
  - **Random Uniform Initialization** (with optional bounds and seed)
  - **Random Normal Initialization** (with optional mean, variance, and seed)
  - **Xavier Initialization**
  - **He Initialization**

## Requirements
- Python 3.x
- Numpy
- Sklearn
- Matplotlib
- json
- networkx

## Setting Up
- Clone this repository on your terminal `git clone https://github.com/pandaandsushi/IF3270_Tubes1_FFNN.git`
- Open Visual Studio Code 

## How To Use
1. Install the required libraries:

    ```bash
    pip install numpy sklearn matplotlib
    ```
2. Run the provided `.ipynb` notebook file and execute the `main()` function in Python:

    ```python
    from main import main
    main(X_train, X_test, y_train, y_test)
    ```

## Contribution
| Names                     | NIM      | Contribution      |
| ----------------------    |:--------:|:-----------------:|
| Thea Josephine Halim      | 13522012 | Code, Report      |
| Raffael Boymian Siahaan   | 13522046 | Code, Report      |