# Stock Prediction using diferent models

## Abstract

This was an experiment to test a Self Modifying Recurrent Neural Network for Stock predictions, but after seeing the poor performance of the concept at network level I started to try out other methos for comparison.

## Data Processing

Most of the testing was made with a EUR to USD dataset with daily records of open close high and low data points, all where normalized and where fed iteratively to the model a number steps (ej: 20 days) and then compared the output of the model with the Open price of the next day.

## SMRNN (Self Modifying Recurrent Neural Network)

The concept of this type of network was to achieve a network that could learn to learn, sort of, because in theory the model would learn to adapt its weights to react diferent to the inputs and reflect it on the state. However testing showed that the network achieved decent results just by freezing the weight change and acting as a basic RNN

The schema the network followed was this one:

![](imgs/SMRNN_diagram.svg)

### Variations

I made 2 variations of SMRNN for testing, SMRNN2 is one that just uses the Net 2 output as weigths and biases for net3, and SMRNN3 that has learnable parameters as Net 3 weights and biases that sum up with the output of net2 to calculate the Net 3 output.

## RNN and FGRNN (Forget Gate Recurrrent Neural Network)

This where implementations of the recurrent neural networks units but at a network level, the first with only two networks ( one for calculating the state and another for the output) and the second one with a third layer to act as a sort of forget gate of the state.

This ones achieved better performance than the SMRNN model but also presented the same Freezing problem with the state (the state remain constant no matter the inputs)

## LSTM (long sort term memory)

A simple naive implementation of a model with one layer with x lstm units and a linear layer to merge all into the final output

## SMLSTM

A modification of a lstm layer that include generated weights that are used for calculating another forget gate to apply to the cells states.

## Hybrid Model

A model that has a GRU layer, a LSTM layer and finally a SMLSTM layer joined with a linear layer to adapt to the number of inputs.

# RESULTS

Self modifying nets in this training at least, just learned to make the weights change constant or zero and learned as a normal RNN. The best results in predicting next day prices was the LSTM and the Hybrid models although results show that the models learn to almost mimic the last price with little trend prediction.

I tested long term learning with this models, the training consisted on a number of passes through price data and then some passes making the model output its following input and compared to the right prices for later backpropagation. results show than models mostly learn to output the same paramaters and more than 5 look ahead steps result in terrible results. The most promising result was the Hybrid model with 15 learning steps and 5 prediction steps.
