# Stock Prediction using diferent models

## Abstract

This was an experiment to test a Self Modifying Recurrent Neural Network for Stock predictions, but after seeing the poor performance of the concept at network level I started to try out other methos for comparison.

## Data Processing

Most of the testing was made with a EUR to USD dataset with daily records of open close high and low data points, all where normalized and where fed iteratively to the model a number steps (ej: 20 days) and then compared the output of the model with the Open price of the next day.

## SMRNN (Self Modifying Recurrent Neural Network)

The concept of this type of network was to achieve a network that could learn to learn, sort of, because in theory the model would learn to adapt its weights to react diferent to the inputs and reflect it on the state. However testing showed that the network achieved decent results just by freezing the weight change and acting as a basic RNN

The schema the network followed was this one:

![](imgs/SMRNN_diagram.svg)

## RNN and FGRNN (Forget Gate Recurrrent Neural Network)

This where implementations of the recurrent neural networks units but at a network level, the first with only two networks ( one for calculating the state and another for the output) and the second one with a third layer to act as a sort of forget gate of the state.

This ones achieved better performance than the SMRNN model but also presented the same Freezing problem with the state (the state remain constant no matter the inputs)

## LSTM (long sort term memory)

A simple naive implementation of a model with one layer with x lstm units and a linear layer to merge all into the final output
