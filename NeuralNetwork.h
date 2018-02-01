#ifndef NEURALNETWORKS_H_
#define NEURALNETWORKS_H_

#include "stdafx.h"
using namespace std;

class Neuron; 

class NeuralNetwork {
private:
	double SENSOR0;						/* Input1 of the Neural Network, normalised */
	double SENSOR1;						/* Input2 of the Neural Network, normalised */
	double left_wheel;					/* Output1 of the Neural Network, normalised */
	double right_wheel;					/* Output2 of the Neural Network, normalised */					

	vector<Neuron*>* input_layer;		/* Array of the Neurons in the Input Layer, (I1 and I2) */
	vector<Neuron*>* hidden_layer;		/* Array of the Neurons in the Hidden Layer, (H1 and H2) */
	vector<Neuron*>* output_layer;		/* Array of the Neurons in the Output Layer, (O1 and O2) */

	double error1;						/* Difference between the desired value for the left wheel and the one calculated */
	double error2;						/* Difference between the desired value for the right wheel and the one calculated */
	double total_error;					/* Error1 + Error2*/

public:
	/* Constructor */
	NeuralNetwork(vector<Neuron*>* il, vector<Neuron*>* hl, vector<Neuron*>* ol);

	/* Getters */
	double getSENSOR0();
	double getSENSOR1();
	double getLeftWheel();
	double getRightWheel();
	vector<Neuron*>* getInputLayer();
	vector<Neuron*>* getHiddenLayer();
	vector<Neuron*>* getOutputLayer();
	double getTotalError();
	double getError1();
	double getError2();

	/* Setters */
	void setSENSOR0(double newValue);
	void setSENSOR1(double newValue);
	void setLeftWheel(double newValue);
	void setRightWheel(double newValue);
	void setTotalError(double error);
	void setError1(double error);
	void setError2(double error);

	/* Logic of the Neural Network */
	void forwardPass();
	void calculatingERROR();
	void backwardPass();
	void UPDATINGweights();
	vector<double> NeuralNetwork::GETweights();
};

#endif