#ifndef CONNECTION_H_
#define CONNECTION_H_

#include <string>
using namespace std;

class Neuron;

class Connection {
private:
	string ID;				/* Connection label (w1, w2, w3, w4, w5, w6, w7, w8, wo=bias) */
	double weight;			/* Weight of the connection, that value will be UPDATED in every iteration */
	double NEWweight = 0;	/* Weight calculated as a result of the backpropagation */
	Neuron* neuron;			/* Neuron that is gonna be input of another one */

public:
	/*Constructor*/
	Connection(string ID, double weight, Neuron* n);

	/* Getters */
	string getID();
	double getWeight();
	double getNEWweight();
	Neuron* getNeuron();

	/* Setters */
	void setWeight(double newWeight);
	void setNEWweight(double newWeight);
};

#endif