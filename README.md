# A Deep Reinforcement Learning algorithm for Swarm Robotics

**Thesis report:** https://github.com/claytonfk/A_DRL_algorithm_for_SR/blob/master/Thesis.pdf

# 1. Requirements

To run the code, the following requirements should be met. The code has not been tested with earlier or later versions of software and libraries described below, thus forward and backward compatibilities are not guaranteed, however, they are expected.

- The simulation software V-REP educational version 3.5.

- A CUDA-enabled graphics processing unit.

- The TensorFlow machine learning library for Python, version 1.3.0 as well as Python programming language, version 3.6.

- Other necessary libraries are matplotlib (2.2.2), numpy (1.14.2) and scipy (1.1.0).

- Windows 64bit, Linux 64bit or Mac OSX operating system.

# 2. Usage

As the first step before running the code - either the training or the evaluation code - it is necessary to execute V-REP and set port 19999 for the communication between the Python code and the software. This can be done by inputting the following command into the console:

```sh
Windows:      start vrep.exe -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE
Linux:       ./vrep.sh -gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE 
Mac:         ./vrep.app/Contents/MacOS/vrep -gREMOTEAPISERVERSERVIC
```

### 2.1 Training

Before running the training file train.py, the user needs to define all options related to the training by editing the file and changing the values of its global variables. Such global variables are described in the following tables. The train.py file can be ran by executing the command "python train.py" in the console.

|       Restore options      |                                                                                   |         |
|:--------------------------:|-----------------------------------------------------------------------------------|---------|
| **Name of global variable**    | **Description**                                                                       | **Type**    |
| restore_model              | Initialize the weights of the network from a saved model                          | boolean |
| restore_em                 | Initialize the experience memory from a saved results file                        | boolean |
| path_to_model_to_restore   | Path to the saved model from which the weights of the network will be restored    | string  |
| path_to_results_to_restore | Path to the results file from which the experience replay memory will be restored | string  |

|       Save options      |                                                                                                            |         |
|:-----------------------:|------------------------------------------------------------------------------------------------------------|---------|
| **Name of global variable** |                                                 **Description**                                                |   **Type**  |
| save_model_frequency    | Frequency (in episodes) of saving models                                                                   | integer |
| max_to_keep             | Limit of models to keep saved in disk. Older models are replaced with new ones once this limit is exceeded | integer |
| path_to_model_to_save   | Path to the model file which will be saved during training                                                 | string  |
| path_to_results_to_save | Path to the results file which will be saved at the end of the training                                    | string  |



|      Stage options      |                                                                               |         |
|:-----------------------:|-------------------------------------------------------------------------------|---------|
| **Name of global variable** |                                                 **Description**                                                |   **Type**  |
| num_agents              | Number of agents used for training                                            | integer |
| num_episodes            | Number of episodes for the stage                                              | integer |
| max_discrepancy         | Discrepancy above which the episode is finished                               | integer |
| min_discrepancy         | Discrepancy below which the episode is finished                               | integer |
| steps_limit             | Number of steps in each episode                                               | integer |
| desired_distance        | Desired distance l in meters                                                  | float   |
| task                    | Task ID.  0: dispersion 1: square formation 2: aggregation 3: chain formation | integer |



| Experience replay memory and Boltzmann exploration and exploitation options |                                    |         |
|:---------------------------------------------------------------------------:|------------------------------------|---------|
| **Name of global variable** |                                                 **Description**                                                |   **Type**  |
| em_capacity                                                                 | Experience replay memory capacity  | integer |
| alpha                                                                       | Exponent alpha                     | float   |
| beta                                                                        | Initial value of the exponent beta | float   |
| final_beta                                                                  | Final value of the exponent beta   | float   |
| initial_b_temperature                                                       | Initial Boltzmann temperature      | float   |
| final_b_temperature                                                         | Final Boltzmann temperature        | float   |


|     Network options     |                                                                                                               |         |
|:-----------------------:|---------------------------------------------------------------------------------------------------------------|---------|
| **Name of global variable** |                                                 **Description**                                                |   **Type**  |
| training_frequency      | Frequency (in time steps) at which the network is trained                                                     | integer |
| batch_size              | Batch size used to train the network                                                                          | integer |
| time_steps              | Number of time steps used for the LSTM cell                                                                   | integer |
| lstm_units              | Number of units of which the LSTM cell is comprised                                                           | integer |
| num_neurons             | Number of neurons of the multi-layer perceptron Example: [50, 50] means two layers containing 50 neurons each | list    |
| copy_weights_frequency  | Frequency (in time steps) at which the weights are copied from online network to target network               | integer |
| discount_factor         | Discount factor of the deep reinforcement learning algorithm                                                  | float   |
| learning_rate           | Learning rate of the optimization algorithm                                                                   | float   |


### 3.2 Evaluation

Similarly, before running the evaluation.py file, it is necessary to edit the file by changing the global variables (table below) according to how the user chooses to perform the evaluation. After this has been done, the evaluation file can be run by executing "python evaluation.py" in the console.


|    Evaluation options    |                                                                                                               |         |
|:------------------------:|:-------------------------------------------------------------------------------------------------------------:|:-------:|
|  **Name of global variable** |                                                  **Description**                                                  |   **Type**  |
| path_to_model_to_restore | Path to the saved model from which the weights of the network will be restored                                | string  |
| time_steps               | Number of time steps used for the LSTM cell                                                                   | integer |
| lstm_units               | Number of units of which the LSTM cell is comprised                                                           | integer |
| num_neurons              | Number of neurons of the multi-layer perceptron Example: [50, 50] means two layers containing 50 neurons each | list    |
| num_agents               | Number of agents used for evaluating                                                                          | integer |
| num_episodes             | Number of episodes for the evaluation                                                                         | integer |
| max_discrepancy          | Discrepancy above which the episode is finished                                                               | integer |
| min_discrepancy          | Discrepancy below which the episode is finished                                                               | integer |
| steps_limit              | Number of steps in each episode                                                                               | integer |
| desired_distance         | Desired distance $l$ in meters                                                                                | float   |
| task                     | Task ID.  0: dispersion 1: square formation 2: aggregation 3: chain formation                                 | integer |
