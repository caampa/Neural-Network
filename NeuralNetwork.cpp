#include "stdafx.h"

#include "NeuralNetwork.h"
#include "Neuron.h"
#include "Connection.h"

#define E (2.7182818284590452353602874713526624977572470937L)
#define ETA 0.5			/* Learning rate, value between 0.0 and 1.0 */
#define ALPHA 1			/* Multiplier of last weight change (momentum), value between 0.0 and n */
#define LAMBDA 1		/* Parameter inside of the logistic function, value between 0.6 and 0.7 */

using namespace std;

NeuralNetwork::NeuralNetwork(vector<Neuron*>* il, vector<Neuron*>* hl, vector<Neuron*>* ol)
	: input_layer(il), hidden_layer(hl), output_layer(ol)
{
}

/* Getters */
double NeuralNetwork::getSENSOR0() {
	return this->SENSOR0;
}
double NeuralNetwork::getSENSOR1() {
	return this->SENSOR1;
}
double NeuralNetwork::getLeftWheel() {
	return this->left_wheel;
}
double NeuralNetwork::getRightWheel() {
	return this->right_wheel;
}
vector<Neuron*>* NeuralNetwork::getInputLayer() {
	return this->input_layer;
}
vector<Neuron*>* NeuralNetwork::getHiddenLayer() {
	return this->hidden_layer;
}
vector<Neuron*>* NeuralNetwork::getOutputLayer() {
	return this->output_layer;
}
double NeuralNetwork::getTotalError() {
	return this->total_error;
}
double NeuralNetwork::getError1() {
	return this->error1;
}
double NeuralNetwork::getError2() {
	return this->error2;
}

/* Setters */
void NeuralNetwork::setSENSOR0(double newValue) {
	this->SENSOR0 = newValue;
}
void NeuralNetwork::setSENSOR1(double newValue) {
	this->SENSOR1 = newValue;
}
void NeuralNetwork::setLeftWheel(double newValue) {
	this->left_wheel = newValue;
}
void NeuralNetwork::setRightWheel(double newValue) {
	this->right_wheel = newValue;
}
void NeuralNetwork::setTotalError(double error) {
	this->total_error = error;
}
void NeuralNetwork::setError1(double error) {
	this->error1 = error;
}
void NeuralNetwork::setError2(double error) {
	this->error2 = error;
}

/* Logic of the Neural Network */
void NeuralNetwork::forwardPass() {
	
	/* Total NET INPUT for the neurons of the INPUT LAYER */
	vector<Neuron*>::iterator it = this->getInputLayer()->begin();
	(*it)->setOut(this->getSENSOR0());
	it++;
	(*it)->setOut(this->getSENSOR1());

	/* Total NET and OUT INPUT for every neuron of the HIDDEN LAYER */
	for (vector<Neuron*>::iterator it = this->getHiddenLayer()->begin(); it != this->getHiddenLayer()->end() ; ++it) {
		double x = 0.0;
		/* Iteration throught every connection of the neuron */
		/* net = i1*w1 + i2*w2 + b1*w0 */
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			x += (*connection)->getNeuron()->getOut() * (*connection)->getWeight();
		}
		/* Net input */
		(*it)->setNet(x);
		/* Out input - Logistic function (1 / (1 + e^-net))*/
		(*it)->setOut(1 / (1 + pow(E, LAMBDA*(-x))));
	}

	/* Total NET and OUT INPUT for every neuron of the OUTPUT LAYER */
	for (vector<Neuron*>::iterator it = this->getOutputLayer()->begin(); it != this->getOutputLayer()->end(); ++it) {
		double x = 0.0;
		/* Iteration throught every connection of the neuron */
		/* net = i1*w1 + i2*w2 + b1*w0 */
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			x += (*connection)->getNeuron()->getOut() * (*connection)->getWeight();
		}
		/* Net input */
		(*it)->setNet(x);
		/* Out input - Logistic function (1 / (1 + e^-net))*/
		(*it)->setOut((1 / (1 + pow(E, LAMBDA*(-x)))));
	}
}

void NeuralNetwork::calculatingERROR() {
	vector<Neuron*>::iterator it = this->getOutputLayer()->begin();
	this->setError1(0.5 * pow((this->left_wheel - (*it)->getOut()), 2));
	it++;
	this->setError2(0.5 * pow((this->right_wheel - (*it)->getOut()), 2));
	this->setTotalError(this->error1 + this->error2);
}


void NeuralNetwork::backwardPass() {
	double Etotal_w = 0.0;		 /* Derivation of the total_error respect a specific weight */
	double Etotal_out = 0.0;	 /* Derivation of the total_error respect the output of the neuron */
	double out_net = 0.0;		 /* Derivation of the output of a neuron respect the input of the same neuron */
	double net_w = 0.0;			 /* Derivation of the input of a neuron respect a specific weight */
	double EO1_out = 0.0;		 /* Derivation of the LEFT WHEEL error */
	double EO2_out = 0.0;		 /* Derivation of the RIGHT WHEEL error */

	/*  Calculating NEW weights for the OUTPUT LAYER */
	for (vector<Neuron*>::iterator it = this->getOutputLayer()->begin(); it != this->getOutputLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			if (((*connection)->getID() != "b1") && ((*connection)->getID() != "b2")) { //Skip the bias
				double w = (*connection)->getWeight();

				/* ETOTAL derivate OUT*/
				/*  Derivation of the total_error respect the output of the neuron */
				/* Comprobation if the neuron is the one that returns the value for the LEFT or RIGHT wheel */
				if ((*it)->getID() == "o1")
					Etotal_out = -(this->left_wheel - (*it)->getOut());
				else if ((*it)->getID() == "o2") Etotal_out = -(this->right_wheel - (*it)->getOut());

				/* OUT derivate NET */
				/* Derivation of the output of a neuron respect the input of the same neuron */
				out_net = ((*it)->getOut() * (1 - (*it)->getOut()));

				/* NET derivate W */
				/* Derivation of the input of a neuron respect a specific weight */
				net_w = ((*connection)->getNeuron()->getOut());

				/* ETOTAL derivate W */
				/* Derivation of the total_error respect a specific weight */
				Etotal_w = Etotal_out * out_net * net_w;

				/* Obtaining the NEW WEIGHT */
				/* W+ = W - (eta * ETOTAL derivate W)*/
				(*connection)->setNEWweight((ALPHA*w) - (ETA * Etotal_w));
			
			}
		}
	}

	/*  Calculating NEW weights for the HIDDEN LAYER */
	for (vector<Neuron*>::iterator it = this->getHiddenLayer()->begin(); it != this->getHiddenLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			if (((*connection)->getID() != "b1") && ((*connection)->getID() != "b2")) { //Skip the bias
				double w = (*connection)->getWeight();

				/* ETOTAL derivate OUT = EO1 derivate OUT + EO2 derivate OUT*/
				/*  Derivation of the total_error respect the output of the neuron */
				double E1_out = 0.0, E2_out = 0.0, out1_net1 = 0.0, out2_net2 = 0.0;
				double  E1_net = 0.0, e1_out = 0.0, E1_netout = 0.0; // E1_net
				double  E2_net = 0.0, e2_out = 0.0, E2_netout = 0.0; // E2_net


				///////////////////////////////////////////////////////////////////////// O1 (Output Neuron 1 - Left wheel)
				// Etotal derivate OUT01
				Neuron* neuron_o1 = (*this->getOutputLayer()->begin());
				e1_out = -(this->left_wheel - neuron_o1->getOut());
				// OUTO1 derivate NETO1
				out1_net1 = neuron_o1->getOut() * (1 - neuron_o1->getOut());
				/* E1 derivate NET */
				E1_net = e1_out * out1_net1;
				/* NET derivate OUT */
				E1_netout = (*(*it)->getOutputConnections()->begin())->getWeight();

				///////////////////////////////////////////////////////////////////////// O2 (Output Neuron 2 - Right wheel)
				Neuron* neuron_o2 = (*(++this->getOutputLayer()->begin()));
				// Etotal derivate OUT02
				e2_out = -(this->right_wheel - neuron_o2->getOut());
				// OUTO2 derivate NETO2
				out2_net2 = neuron_o2->getOut() * (1 - neuron_o2->getOut());
				/* E2 derivate NET */
				E2_net = e2_out * out2_net2;
				/* NET derivate OUT */
				E2_netout = (*++(*it)->getOutputConnections()->begin())->getWeight();

				/* E1 derivate OUT = E1 derivate NET * NET derivate OUT */
				E1_out = E1_net * E1_netout;
				/* E2 derivate OUT = E2 derivate NET * NET derivate OUT */
				E2_out = E2_net * E2_netout;
				/* ETOTAL derivate OUT = EO1 derivate OUT + EO2 derivate OUT*/
				Etotal_out = E1_out + E2_out;


				/* OUT derivate NET */
				/* Derivation of the output of a neuron respect the input of the same neuron */
				out_net = ((*it)->getOut() * (1 - (*it)->getOut()));


				/* NET derivate W */
				/* Derivation of the input of a neuron respect a specific weight */
				net_w = ((*connection)->getNeuron()->getOut());


				/* ETOTAL derivate W */
				/* Derivation of the total_error respect a specific weight */
				Etotal_w = Etotal_out * out_net * net_w;

				/* Obtaining the NEW WEIGHT */
				/* W+ = W - (eta * ETOTAL derivate W)*/
				(*connection)->setNEWweight((ALPHA*w) - (ETA * Etotal_w));
			}
		}
	}
}

void NeuralNetwork::UPDATINGweights() {

	/*  Updating ALL the weights */
	for (vector<Neuron*>::iterator it = this->getHiddenLayer()->begin(); it != this->getHiddenLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			if (((*connection)->getID() != "b1") && ((*connection)->getID() != "b2")) //The wieght of the bias doesn't have to update
			(*connection)->setWeight((*connection)->getNEWweight());
		}
	}
	for (vector<Neuron*>::iterator it = this->getOutputLayer()->begin(); it != this->getOutputLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			if (((*connection)->getID() != "b1") && ((*connection)->getID() != "b2")) //The wieght of the bias doesn't have to update
				(*connection)->setWeight((*connection)->getNEWweight());
		}
	}
}

vector<double> NeuralNetwork::GETweights() {

	vector<double> weights;

	for (vector<Neuron*>::iterator it = this->getHiddenLayer()->begin(); it != this->getHiddenLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			weights.push_back((*connection)->getWeight());
		}
	}
	for (vector<Neuron*>::iterator it = this->getOutputLayer()->begin(); it != this->getOutputLayer()->end(); ++it) {
		for (vector<Connection*>::iterator connection = (*it)->getInputConnections()->begin(); connection != (*it)->getInputConnections()->end(); ++connection) {
			weights.push_back((*connection)->getWeight());
		}
	}
	return weights;
}