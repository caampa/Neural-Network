# Neural-Network
System that allows a Pioneer robot to perform a Left Edge Following implementing from scratch a [Neural Network](./NN_graph.png).

## Organization of the folders
. It is impotant to mention the content of the following folders:
- **Root** *All the code is given in the root folder. Also, this folder contains the training data set recollected from a Pioneer robot that performs a left-edge-following. This file has four columns (1º: Distance between the sensor zero of the robot and the wall, 2º Distance between the sensor 1 and the wall, 3º speed for the left wheel of the robot and 4º speed for the right wheel of the robot).*
- **Files** *It gathers information about the performance of the final Neural Network. Illustrating the Error for every row in the training file in all the 20 "Epoch" in which the NN was trained, the Root Mean Square Error (RMSE) of each "Epoch" and storing the minimum RMSE next to its *optimal weights* the content generated in this folder is highly relevant in order to know which weights should be chosen.*
- **Graphs** *It shows the graphs relating to the training, validation and test results. Quite useful for the stopping criteria.*

## Structure of the system
The structure of the system can be seen in the [` NN_classdiagram.png `](./NN_classdiagram.png) image.

Basically, this project has two different parts: the first one, in which a NN is built and trained and the second part, that allows to a Pioneer robot to use this NN already built in order to be capable to perform a left edge following.

### Training and building the NN
For the first part, the code below need to be uncommented from the ` NeuralNetwork_ml17390.cpp ` file:

``` 
	/* NEURAL NETWORK CONFIGURATION(TRAINING) */
	/* Training Nerural Network to calculate the optimal weghts */
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
	writeMinRMSE(min_rmse, min_val_rmse);
	
```

In this part, the connection with the robot is not necessary yet. In addition, the training and validation files are specified in the ` trainNeuralNetwork ` and ` validateNeuralNetwork ` methods. Also, if those files are modified, it should be convenient to change the maximun and minimum distances and velocities in the method ` normalizeValues `.

> It is very normal for this process to spend more than one hour of training. The number of "Epoch" and features of the computer will determinate it.

### Using the NN
 The above code needs to be commented again. Now, the NN has given new optimal weights, setting them up is quite easy. 
 The method ` buildNeuralNetwork ` has the following code where they will be changed:
 
 ``` 
 /* Building ALL the connections between the neurons */
	Connection* w1 = new Connection("w1", -4.240379588625, i1); 
	Connection* w2 = new Connection("w2", -0.15458900033383655, i2);
	Connection* w3 = new Connection("w3", 2.4517917984212381, i1); 
	Connection* w4 = new Connection("w4", 4.1600608878887932, i2); 
	Connection* b1 = new Connection("b1", 0.35, bias1); 
	Connection* w5 = new Connection("w5", 4.1552896885490691, h1); 
	Connection* w6 = new Connection("w6", -3.6080522477942036, h2); 
	Connection* w7 = new Connection("w7", -7.9632027387639281, h1); 
	Connection* w8 = new Connection("w8", 1.8398501908642131, h2); 
	Connection* b2 = new Connection("b2", 0.60, bias2); 
  
```

Then, the NN is ready. In the code below can be seen how the readings from the robot are integrated with the NN, how the NN is able to predict new speeds for both wheel using the two given distances obtained from the sensors zero and one.

```
while (true) {

		/*Recollection of the sonar distances*/
		for (int i = 0; i < NUM_SENSORS; i++) {
			sonar_sensor[i] = robot.getSonarReading(i);
			sonar_range[i] = sonar_sensor[i]->getRange();
		}
		/* NEURAL NETWORK PREDICTION */
		/* Predicting NEW speed for the wheels */
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
	}
	
```

Acknowledgments to Matt Mazur for given such as good example on how a Neural Network works available in (https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
  
