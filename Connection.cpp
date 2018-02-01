#include "stdafx.h"
#include "Connection.h"
using namespace std;

Connection::Connection(string id, double weight, Neuron* n)
	: ID(id), weight(weight), neuron(n)
{
}

/* Getters */
string Connection::getID() {
	return this->ID;
}
double Connection::getWeight() {
	return this->weight;
}
double Connection::getNEWweight() {
	return this->NEWweight;
}
Neuron* Connection::getNeuron() {
	return this->neuron;
}

/* Setters */
void Connection::setWeight(double newWeight) {
	this->weight = newWeight;
}
void Connection::setNEWweight(double newWeight) {
	this->NEWweight = newWeight;
}


