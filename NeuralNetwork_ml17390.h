#pragma once

#include "Connection.h"
#include "Neuron.h"
#include "NeuralNetwork.h"

#define NUM_SENSORS 2
#define NUM_EPOCH 10   //100	/* One forward pass and one backward pass of all the training examples */

/** GLOBAL VARIABLE **/
ArRobot robot;

/** ROBOT CONFIGURATION **/
void ROBOTconfiguration();
void ROBOTshutdown();

/** NEURAL NETWORK CONFIGURATION **/
NeuralNetwork* buildNeuralNetwork();
void iterateNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors, int iteration_number, double s0, double s1, double lw, double rw);
void normalizeValues(double* s0, double* s1, double* lw, double* rw);
void trainNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors);
void validateNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors);
void denormalizeValues(double* lw, double* rw);
void useNeuralNetwork(NeuralNetwork* nn, vector<double>* error, double s0, double s1, double lw, double rw);

/** DATA ANALYSIS **/
void writeErrors(int num_epoch, map<int, vector<double>>* rmse);
void rootMeanSquaredError(NeuralNetwork* nn, int num_epoch, map<int, vector<double>>* rmsq, map<int, vector<double>>* errors);
vector<double> minRMSE(map<int, vector<double>>* rmse);
void writeMinRMSE(vector<double> rmse, vector<double> v_rmse);
void writeRMSE(map<int, vector<double>>* rmse);
void collectDataFromRobot(double s0, double s1, double lw, double rw);