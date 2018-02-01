#ifndef NEURON_H_
#define NEURON_H_

#include <vector>
#include <string>
using namespace std;

class Connection;

class Neuron {
private:
	string ID;										/* Neuron label (i1,i2,h1,h2,o1,o2) */
	double net = 0.0;								/* Total net input of a neuron */
	double out = 1.0;								/* Real output of a neuron, value after calculate the activation function */ //1 by default for the BIAS
	vector<Connection*>* inputConnections;			/* Array of weights that a specific neuron receibe as an input */
	vector<Connection*>* outputConnections;			/* Array of weights that a specific neuron receibe as an output */

public:
	/* Constructor */
	Neuron(string id);

	/* Getters */
	string getID();
	double getNet();
	double getOut();
	vector<Connection*>* getInputConnections();
	vector<Connection*>* getOutputConnections();

	/* Setters */
	void setNet(double newNet);
	void setOut(double newOut);
	void configInputConnections(vector<Connection*>* ic);
	void configOutputConnections(vector<Connection*>* ic);

};

#endif