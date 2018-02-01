#include "stdafx.h"
#include "Neuron.h"
#include "Connection.h"

using namespace std;

Neuron::Neuron(string id)
	:ID(id)
{}

/* Getters */
string Neuron::getID() {
	return this->ID;
}
double Neuron::getNet() {
	return this->net;
}
double  Neuron::getOut() {
	return this->out;
}
vector<Connection*>* Neuron::getInputConnections() {
	return this->inputConnections;
}

vector<Connection*>* Neuron::getOutputConnections() {
	return this->outputConnections;
}

/* Setters */
void  Neuron::setNet(double net) {
	this->net = net;
}
void  Neuron::setOut(double out) {
	this->out = out;
}
void Neuron::configInputConnections(vector<Connection*>* ic) {
	this->inputConnections = ic;
}
void Neuron::configOutputConnections(vector<Connection*>* oc) {
	this->outputConnections = oc;
}