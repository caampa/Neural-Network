// NeuralNetwork_ml17390.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "NeuralNetwork_ml17390.h"


/** MAIN **/
int main(int argc, char **argv)
{
	/* ROBOT CONNECTION */
	Aria::init();
	ArPose pose;
	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		cout << "Robot Connected! --------------------- :)" << endl;

	/* ROBOT CONFIGURATION */
	ROBOTconfiguration();

	/* SENSORS CONFIGURATION */
	int sonar_range[NUM_SENSORS];
	ArSensorReading *sonar_sensor[NUM_SENSORS];

	/* NEURAL NETWORK INITIALIZATION */
	NeuralNetwork* nn = buildNeuralNetwork();
	vector<double>* error = new vector<double>();

	/* NEURAL NETWORK CONFIGURATION(TRAINING) */
	/* Training Nerural Network to calculate the optimal weghts *
	map<int, vector<double>>* train_errors = new map<int, vector<double>>();
	map<int, vector<double>>* train_rmse = new map<int, vector<double>>();

	map<int, vector<double>>* val_errors = new map<int, vector<double>>();
	map<int, vector<double>>* val_rmse = new map<int, vector<double>>();

	for (int i = 0; i < 20; i++) {
		trainNeuralNetwork(nn, train_errors);
		validateNeuralNetwork(nn, val_errors);
		rootMeanSquaredError(nn, i, train_rmse, train_errors);
		rootMeanSquaredError(nn, i, val_rmse, val_errors);
		writeErrors(i,val_errors);
	}
	//writeRMSE(train_rmse);
	vector<double> min_rmse = minRMSE(train_rmse);
	vector<double> min_val_rmse = minRMSE(val_rmse);
	writeMinRMSE(min_rmse, min_val_rmse);*/

	while (true) {

		/*Recollection of the sonar distances*/
		for (int i = 0; i < NUM_SENSORS; i++) {
			sonar_sensor[i] = robot.getSonarReading(i);
			sonar_range[i] = sonar_sensor[i]->getRange();
		}
		/* NEURAL NETWORK PREDICTION */
		/* Predicting NEW speed for the wheels */

		// Trying with these values: 574.75, 2851.29, 115, 147.054
		// Values predicted: 136.89057872581580, 138.39490806190156
		// The total error = 0.057186988239066296, error1 = 0.056155479340572106, error2 = 0.0010315088984941877
		// Trying with these values:  377.423, 2352.4, 125.442, 115
		// Values predicted: 142.29338068811791, 129.45430496056554
		// The total error = 0.036151579982004640, error1 = 0.033277342133807356, error2 = 0.0028742378481972871
		double s0 = sonar_range[0], s1 = sonar_range[1];
		useNeuralNetwork(nn, error, sonar_range[0], sonar_range[1], 0.0, 0.0); //The two last arguments are just for test

		/* Converting the NEW values to the according velocity */
		double left_wheel = nn->getLeftWheel();
		double right_wheel = nn->getRightWheel();
		denormalizeValues(&left_wheel, &right_wheel);

		cout << "Sensor0: " << sonar_range[0] << " - " << " Sensor1: " << sonar_range[1] << endl;
		cout << "Left: " << left_wheel << " - " << " Right: " << right_wheel << endl;

		/* ROBOT SETTING NEW SPEED */
		robot.setVel2(left_wheel, right_wheel);

		/* DATA COLLECTION FOR 20 SECONDS IN THE REAL ROBOT */
		collectDataFromRobot(s0, s1, left_wheel, right_wheel);

	}

	/* ROBOT SHUTTING DOWN */
	ROBOTshutdown();

	return 0;
}


/** ROBOT CONFIGURATION **/

void ROBOTconfiguration() {
	/**/
	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();
}

void ROBOTshutdown() {
	robot.lock();
	robot.stop();
	robot.unlock();
	Aria::exit();
}

/** NEURAL NETWORK CONFIGURATION **/

NeuralNetwork* buildNeuralNetwork() {

	/* Building the neurons of the INPUT LAYER */
	Neuron* i1 = new Neuron("i1");
	Neuron* i2 = new Neuron("i2");
	Neuron* bias1 = new Neuron("b1");
	/* Building the neurons of the HIDDEN LAYER */
	Neuron* h1 = new Neuron("h1");
	Neuron* h2 = new Neuron("h2");
	Neuron* bias2 = new Neuron("b2");
	/* Building the neurons of the OUTPUT LAYER */
	Neuron* o1 = new Neuron("o1");
	Neuron* o2 = new Neuron("o2");

	/* Building ALL the connections between the neurons */
	Connection* w1 = new Connection("w1", -4.240379588625, i1); //0.15 // -4.240379588625 // 
	Connection* w2 = new Connection("w2", -0.15458900033383655, i2); //0.20 // -0.15458900033383655 //
	Connection* w3 = new Connection("w3", 2.4517917984212381, i1); // 0.25 // 2.4517917984212381 // 
	Connection* w4 = new Connection("w4", 4.1600608878887932, i2); //0.30 //  4.1600608878887932 //
	Connection* b1 = new Connection("b1", 0.35, bias1); //0.35 // 0.35 // 
	Connection* w5 = new Connection("w5", 4.1552896885490691, h1); //0.40 //  4.1552896885490691 // 
	Connection* w6 = new Connection("w6", -3.6080522477942036, h2); //0.45 // -3.6080522477942036 //
	Connection* w7 = new Connection("w7", -7.9632027387639281, h1); //0.50 // -7.9632027387639281 //
	Connection* w8 = new Connection("w8", 1.8398501908642131, h2); //0.55 // 1.8398501908642131 // 
	Connection* b2 = new Connection("b2", 0.60, bias2); //0.60 // 0.60 // 

	/* i1 */
	vector<Connection*>* i1c = new vector<Connection*>();
	i1c->push_back(w1);
	i1c->push_back(w3);
	i1->configOutputConnections(i1c);
	/* i2 */
	vector<Connection*>* i2c = new vector<Connection*>();
	i2c->push_back(w2);
	i2c->push_back(w4);
	i2->configOutputConnections(i2c);
	/* h1 */
	vector<Connection*>* h1c1 = new vector<Connection*>();
	h1c1->push_back(w1);
	h1c1->push_back(w2);
	h1c1->push_back(b1);
	h1->configInputConnections(h1c1);
	vector<Connection*>* h1c2 = new vector<Connection*>();
	h1c2->push_back(w5);
	h1c2->push_back(w7);
	h1->configOutputConnections(h1c2);
	/* h2 */
	vector<Connection*>* h2c1 = new vector<Connection*>();
	h2c1->push_back(w3);
	h2c1->push_back(w4);
	h2c1->push_back(b1);
	h2->configInputConnections(h2c1);
	vector<Connection*>* h2c2 = new vector<Connection*>();
	h2c2->push_back(w6);
	h2c2->push_back(w8);
	h2->configOutputConnections(h2c2);
	/* o1 */
	vector<Connection*>* o1c = new vector<Connection*>();
	o1c->push_back(w5);
	o1c->push_back(w6);
	o1c->push_back(b2);
	o1->configInputConnections(o1c);
	/* o2 */
	vector<Connection*>* o2c = new vector<Connection*>();
	o2c->push_back(w7);
	o2c->push_back(w8);
	o2c->push_back(b2);
	o2->configInputConnections(o2c);

	/* INPUT LAYER NEURONS */
	vector<Neuron*>* il = new vector<Neuron*>();
	il->push_back(i1);
	il->push_back(i2);

	/* HIDDEN LAYER NEURONS */
	vector<Neuron*>* hl = new vector<Neuron*>();
	hl->push_back(h1);
	hl->push_back(h2);

	/* OUTPUT LAYER NEURONS */
	vector<Neuron*>* ol = new vector<Neuron*>();
	ol->push_back(o1);
	ol->push_back(o2);

	NeuralNetwork* nn = new NeuralNetwork(il, hl, ol);
	return nn;
}

void normalizeValues(double* s0, double* s1, double* lw, double* rw) {
	//To safe operations on the computer, the minimun and maximun value of each variable has been calculated in excel sorting the columns
	// $normalize = ($denormalize - $min)/($max - $min)
	/* SENSOR0 */
	double min_s0 = 258.765;
	double max_s0 = 6072.12;
	(*s0) = ((*s0) - min_s0) / (max_s0 - min_s0);
	/* SENSOR1 */
	double min_s1 = 922.15;
	double max_s1 = 4410.45;
	(*s1) = ((*s1) - min_s1) / (max_s1 - min_s1);
	/* LEFT WHEEL SPEED */
	double min_lw = 115;
	double max_lw = 180.32;
	(*lw) = ((*lw) - min_lw) / (max_lw - min_lw);
	/* RIGHT WHEEL SPEED */
	double min_rw = 109.357;
	double max_rw = 300;
	(*rw) = ((*rw) - min_rw) / (max_rw - min_rw);
}

void trainNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors) {
	ifstream file("all_data1.txt");
	string line;
	double s0, s1, lw, rw;
	//double s0 = 0.05, s1 = 0.10, lw = 0.01, rw = 0.99;
	int iteration_number = 0;

	//Reading from the training data file
	while (getline(file, line)) {

		/* Spliting the line and getting the values */
		istringstream values(line);
		char coma;
		values >> s0 >> coma >> s1 >> coma >> lw >> coma >> rw;

		/* Normalize values (Transform the numbers into another number between 0 and 1)*/
		normalizeValues(&s0, &s1, &lw, &rw);

		/* Training the Neural Networks with the NEW values from the training data */
		iterateNeuralNetwork(nn, errors, iteration_number, s0, s1, lw, rw);
		iteration_number++;
	}
}

void validateNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors) {
	ifstream file("all_data2.csv");
	string line;
	double s0, s1, lw, rw;
	//double s0 = 0.05, s1 = 0.10, lw = 0.01, rw = 0.99;
	int iteration_number = 0;

	//Reading from the training data file
	while (getline(file, line)) {

		/* Spliting the line and getting the values */
		istringstream values(line);
		char coma;
		values >> s0 >> coma >> s1 >> coma >> lw >> coma >> rw;

		/* Normalize values (Transform the numbers into another number between 0 and 1)*/
		normalizeValues(&s0, &s1, &lw, &rw);

		/* Training the Neural Networks with the NEW values from the training data */
		iterateNeuralNetwork(nn, errors, iteration_number, s0, s1, lw, rw);
		iteration_number++;
	}
}

void iterateNeuralNetwork(NeuralNetwork* nn, map<int, vector<double>>* errors, int iteration_number, double s0, double s1, double lw, double rw) {

	/* Passing new values to the neuron */
	nn->setSENSOR0(s0);
	nn->setSENSOR1(s1);
	nn->setLeftWheel(lw);
	nn->setRightWheel(rw);
	/* Forward propagation of the neuron */
	nn->forwardPass();
	/* Error respect the desired values and the values passed aboved to the neuron*/
	nn->calculatingERROR();
	/* Backward propagation of the neuron */
	nn->backwardPass();
	/* Setting the weights with the values predicted above */
	nn->UPDATINGweights();

	/* Collecting: the TOTAL ERROR, the ERROR of the LEFT wheel and the ERROR of the RIGHT wheel for every iteration*/
	vector<double> e;
	e.push_back((nn->getError1() + nn->getError2()) / 2); //Mean
	e.push_back(nn->getError1());
	e.push_back(nn->getError2());
	(*errors)[iteration_number] = e;

}

void denormalizeValues(double* lw, double* rw) {
	//To safe operations on the computer, the minimun and maximun value of each variable has been calculated in excel sorting the columns
	//$denormalize = ($normalized * ($max - $min) + $min);
	/* LEFT WHEEL SPEED */
	double min_lw = 115;
	double max_lw = 180.32;
	(*lw) = (*lw) * (max_lw - min_lw) + min_lw;
	/* RIGHT WHEEL SPEED */
	double min_rw = 109.357;
	double max_rw = 300;
	(*rw) = (*rw) * (max_rw - min_rw) + min_rw;
}

void useNeuralNetwork(NeuralNetwork* nn, vector<double>* error, double s0, double s1, double lw, double rw) {

	/* Normalize values (Transform the numbers into another number between 0 and 1)*/
	normalizeValues(&s0, &s1, &lw, &rw);

	/* Sending new values to the neuron */
	nn->setSENSOR0(s0);
	nn->setSENSOR1(s1);
	//nn->setLeftWheel(lw);  //Just to be able to calculate the error afterwards
	//nn->setRightWheel(rw); //Just to be able to calculate the error afterwards

	/* Forward propagation of the neuron */
	nn->forwardPass();
	/* Error respect the desired values and the values passed aboved to the neuron*/
	//nn->calculatingERROR();
	/* Setting the values predicted to the wheels */
	nn->setLeftWheel((*(nn->getOutputLayer()->begin()))->getOut());
	nn->setRightWheel((*(++(nn->getOutputLayer()->begin())))->getOut());

	/* Collecting: the TOTAL ERROR, the ERROR of the LEFT wheel and the ERROR of the RIGHT wheel for the prediction (Just for testing) */
	//error->push_back(nn->getTotalError());
	//error->push_back(nn->getError1());
	//error->push_back(nn->getError2());

}

/** DATA ANALYSIS **/

void writeErrors(int num_epoch, map<int, vector<double>>* errors) {
	ofstream myfile;

	/* Average */
	string file1 = "Files\\error_average_epoch_" + to_string(num_epoch) + ".txt";
	myfile.open(file1);
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[0] << endl;
	}
	myfile.close();

	/* Error1 */
	string file2 = "Files\\error1_epoch_" + to_string(num_epoch) + ".txt";
	myfile.open(file2);
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[1] << endl;
	}
	myfile.close();

	/* Error2 */
	string file3 = "Files\\error2_epoch_" + to_string(num_epoch) + ".txt";
	myfile.open(file3);
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[2] << endl;
	}
	myfile.close();
}

void rootMeanSquaredError(NeuralNetwork* nn, int num_epoch, map<int, vector<double>>* rmse_final, map<int, vector<double>>* errors) {
	double average_te = 0.0, average_e1 = 0.0, average_e2 = 0.0;
	double rmse_te = 0.0, rmse_e1 = 0.0, rmse_e2 = 0.0;
	vector<double> rmse;

	//Average
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		average_te += it->second[0];
		average_e1 += it->second[1];
		average_e2 += it->second[2];
	}
	average_te = (average_te / errors->size());
	average_e1 = (average_e1 / errors->size());
	average_e2 = (average_e2 / errors->size());

	//RMSE
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		rmse_te += pow((it->second[0] - average_te), 2);
		rmse_e1 += pow((it->second[1] - average_e1), 2);
		rmse_e2 += pow((it->second[2] - average_e2), 2);
	}
	/* ROOT MEAN SQUARE ERROR for this epoch */
	rmse.push_back(sqrt(rmse_te / errors->size())); //TOTAL ERROR
	rmse.push_back(sqrt(rmse_e1 / errors->size())); //ERROR LEFT WHEEL
	rmse.push_back(sqrt(rmse_e2 / errors->size())); //ERROR RIGHT WHEEL

													/* LAST WEIGHTS for this epoch */
	rmse.push_back(nn->GETweights()[0]); //W1
	rmse.push_back(nn->GETweights()[1]); //W2
	rmse.push_back(nn->GETweights()[2]); //b1
	rmse.push_back(nn->GETweights()[3]); //W3
	rmse.push_back(nn->GETweights()[4]); //W4
	rmse.push_back(nn->GETweights()[6]); //W5
	rmse.push_back(nn->GETweights()[7]); //b6
	rmse.push_back(nn->GETweights()[8]); //b2
	rmse.push_back(nn->GETweights()[9]); //W7
	rmse.push_back(nn->GETweights()[10]); //W8

	(*rmse_final)[num_epoch] = rmse;

}

vector<double> minRMSE(map<int, vector<double>>* rmse) {
	vector<double> minimun(2);
	minimun[0] = 1; //Min value of the root mean square error average
	minimun[1] = 0;  //Iterator
	for (map<int, vector<double>>::iterator it = rmse->begin(); it != rmse->end(); ++it) {
		if (it->second[0] < minimun[0]) {
			minimun[0] = it->second[0];
			minimun[1] = it->first;
		}
	}
	return (*rmse)[minimun[1]];
}

void writeRMSE(map<int, vector<double>>* errors) {
	ofstream myfile;

	/* Data to plot in charts */
	/* RMSE- Average */
	myfile.open("Files\\root_mean_squared_error_AVERAGE.txt");
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[0] << endl;
	}
	myfile.close();

	/* RMSE- Error1 */
	myfile.open("Files\\root_mean_squared_error_Error1.txt");
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[1] << endl;
	}

	myfile.close();
	/* RMSE- Error2 */
	myfile.open("Files\\root_mean_squared_error_Error2.txt");
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << it->second[2] << endl;
	}
	myfile.close();

	/* Data with Weights - More information */
	myfile.open("Files\\root_mean_squared_error.txt");

	myfile << "Writing: " << "root_mean_squared_error.txt" << endl;
	for (map<int, vector<double>>::iterator it = errors->begin(); it != errors->end(); ++it) {
		myfile << "-----------------------------------------------------------";
		myfile << "Iteration number: " << it->first << endl;
		myfile << "Average: " << it->second[0] << endl;
		myfile << "Error1: " << it->second[1] << endl;
		myfile << "Error2: " << it->second[2] << endl;
		myfile << "-----------------------------------------------------------";
		myfile << "WEIGHTS: " << endl;
		myfile << "W1: " << it->second[3] << endl;
		myfile << "W2: " << it->second[4] << endl;
		myfile << "b1: " << it->second[5] << endl;
		myfile << "W3: " << it->second[6] << endl;
		myfile << "W4: " << it->second[7] << endl;
		myfile << "W5: " << it->second[8] << endl;
		myfile << "W6: " << it->second[9] << endl;
		myfile << "b2: " << it->second[10] << endl;
		myfile << "W7: " << it->second[11] << endl;
		myfile << "W8: " << it->second[12] << endl;
	}

	myfile.close();
}

void writeMinRMSE(vector<double> rmse, vector<double> val_rmse) {
	ofstream myfile, myfile2;

	/* EPOCH that has the min root square error value */
	string file1 = "Files\\min_RMSE.txt";
	myfile.open(file1);
	/*for (vector<double>::iterator it = rmse.begin(); it != rmse.end(); ++it) {
	myfile << *it << endl;
	}*/
	myfile << "Average: " << rmse[0] << endl;
	myfile << "Error1: " << rmse[1] << endl;
	myfile << "Error2: " << rmse[2] << endl;
	myfile << "-----------------------------------------------------------";
	myfile << "WEIGHTS: " << endl;
	myfile << "W1: " << rmse[3] << endl;
	myfile << "W2: " << rmse[4] << endl;
	myfile << "b1: " << rmse[5] << endl;
	myfile << "W3: " << rmse[6] << endl;
	myfile << "W4: " << rmse[7] << endl;
	myfile << "W5: " << rmse[8] << endl;
	myfile << "W6: " << rmse[9] << endl;
	myfile << "b2: " << rmse[10] << endl;
	myfile << "W7: " << rmse[11] << endl;
	myfile << "W8: " << rmse[12] << endl;
	myfile.close();


	/** VALIDATION **/
	/* EPOCH that has the min root square error value */
	string file2 = "Files\\min_val_RMSE.txt";
	myfile2.open(file2);
	myfile2 << "Average: " << val_rmse[0] << endl;
	myfile2 << "Error1: " << val_rmse[1] << endl;
	myfile2 << "Error2: " << val_rmse[2] << endl;
	myfile2 << "-----------------------------------------------------------";
	myfile2 << "WEIGHTS: " << endl;
	myfile2 << "W1: " << val_rmse[3] << endl;
	myfile2 << "W2: " << val_rmse[4] << endl;
	myfile2 << "b1: " << val_rmse[5] << endl;
	myfile2 << "W3: " << val_rmse[6] << endl;
	myfile2 << "W4: " << val_rmse[7] << endl;
	myfile2 << "W5: " << val_rmse[8] << endl;
	myfile2 << "W6: " << val_rmse[9] << endl;
	myfile2 << "b2: " << val_rmse[10] << endl;
	myfile2 << "W7: " << val_rmse[11] << endl;
	myfile2 << "W8: " << val_rmse[12] << endl;
	myfile2.close();
}

void collectDataFromRobot(double s0, double s1, double lw, double rw) {
	ofstream myfile;

	string file = "Files\\data_ROBOT.txt";
	myfile.open(file, ios_base::out);
	myfile << s0 << " " << s1 << " " << lw << " " << rw << endl;
	myfile.close();
}