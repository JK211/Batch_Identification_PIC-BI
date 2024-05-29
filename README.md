# PIC-BI: Practical and Intelligent Combinatorial Batch Identification for UAV assisted IoT Networks
*Keywords: UAV-aided IoT Networks, Combinatorial Framework, Batch Identification, Kalman Filter, Reinforcement Learning*

# Description

The code in the repository implements the combinatorial batch identification framework mentioned in the paper, which specifically contains two algorithms.

- Adaptive Estimation Algorithm for Batch Illegal Ratio (AEABIR) that corresponds to the contribution 1 in the manuscript.
- Combinatorial Strategy Learning algorithm (CSLA) that corresponds to the contribution 2 in the manuscript.

We will explain how to verify the two contribution points using codes. These codes can directly output experimental plots in a short time to make it easier for others to verify the results of our paper. It is worth noting that there are detailed comments in the code explaining the purpose of the code. For example, we indicate which figure in the paper those outputs correspond to. 

Note that our algorithm involves sampling, which is stochastic, and that the paper's data is an average of multiple counts. It is normal for the output of the program to be a little different from the results of the paper, but the trend and magnitude of the data should be the same.

# Overview

- Environment Installation (compiler + dependency packages) [30 human-minutes for newcomer or 10 human-minutes for Reinforcement Learning Practitioner]
- Fully Automated Running Experiments and Verifying Results [10 human-minutes + 20 computer-minutes]
- Train A New Model and Reuse beyond paper [A longer period of time, perhaps dozens of hours]
- An Experiment Showing Video [3 human-minutes]

# Environment Installation

We recommend using the Pycharm compiler; our code involves a lot of reinforcement learning stuff, all in Python, and it should be easier for reinforcement learning practitioners to reproduce our code.

Python Dependencies:

- python 3.7.13
- tate-bilinear-pairing 0.3
- statsmodels 0.13.5
- pytorch 1.10.1
- gym 0.24.0
- numpy 1.21.5
- pandas 1.3.5
- matplotlib 3.5.2

We recommend installing the above version, and utilizing conda/pip for unified management. Sometimes inconsistent dependency versions can trigger some errors.

# Fully Automated Running Experiments and Verifying Results

Run *Estimation.py* can verify Adaptive Estimation Algorithm for Batch Illegal Ratio (AEABIR) for contribution 1, the plt.show() in the code can output the figures automatically. The output of the experiments will validate the following claims:

- Figure 3: the output of code line 215-251 can reproduce Fig.3(a)(b)(c) on page 10.
- Figure 4(a): the output of code line 287-316 can reproduce Fig.4(a) on page 11.
- Figure 4(b): the output of code line 194-211 can reproduce Fig.4(b) on page 11.
- Figure 4(c): the output of code line 267-275 can reproduce Fig.4(c) on page 11.

Run *Efficiency comparision.py* can verify Combinatorial Strategy Learning algorithm (CSLA) for contribution 2, the plt.show() in the code can output the figures automatically. Despite trying to train a unified model, the 300-x model provided for quick code verification is still imperfect. The output of the unified model appears to converge to suboptimal or poorer solutions at less than 20. However, this does not mean that our proposed combinatorial batch identification algorithm is less efficient than existing work. Hence, we give the code *Supplementary Comparision.py*  for verification.

Figure 6(a): the output of code line 185-195 can reproduce Fig.6(a) on page 11.

Run *Supplementary Comparision.py* can verify Combinatorial Strategy Learning algorithm (CSLA) at 300-1, 300-10, 300-20 for contribution 2, the plt.show() in the code can output the figures automatically. 

Run *crypto_cost.py* can verify the costs of cryptographic key operations. Run *inference_delay* can verify the one-time model inference delay.

- Table 4: The experimental results output from the command line correspond to the two cryptographic operations Pairing operation and Scalar multiplication operation in the table. However, it is worth noting that the data in the paper comes from a prototype system based on Raspberry Pi. The output of this code on the laptop is only for reference. The numerical magnitude should be the same, but there is definitely a deviation.

Our artifact does not validate the following claims:

- Figure 6(b): The reason why Fig.6(a) can be easily verified is that we trained a complete 300-x model, but to be honest, our model had some flaws when training 2000-x. The model was incomplete and difficult to conduct simple verification. If you are interested, you can follow our instructions to train your own model for further verification.
- Figure 7: On page 12, we give more experiments from a prototype system based on Raspberry Pi. This cannot be validated without access to specialized hardware, so we leave it out of scope of artifact evaluation. However, the experimental results of Fig6(a) are enough to show that the CSLA algorithm we trained has advantages over other algorithms. Figure 7 is just to further verify the superiority of our proposed solution in a real environment.

# Train A New Model and Reuse beyond paper

- *Batch access simulator.py*: This code is noted for generating batch requests containing illegal requests.
- *Batch_Signature_\*actions.py*: This code is used to build a batch identification environment. It is inherited from Gym and can be used to handle actions, state transitions and reward functions during batch identification.
- *Main_\*-x.py*: This code, together with *Batch access simulator.py*, *Batch_Signatures_\*actions.py* and *DQN.py*, implements combinatorial strategy learning algorithm (CSLA), which can be used to learn and train a model for combinatorial algorithmic identification.

It is possible to train your own model using these three codes, and there are two ways to use them during our research.

- The first one: using a particular batch of requests for learning a combinatorial batch identification algorithm. We have already set up the relevant data, and we can directly run *Main_300-x.py* to learn the combinatorial batch identification algorithm for a batch request of 300-20. 300-20means that there are 20 illegal requests contained in 300 requests. Meanwhile, *BSI.py* and *MRI.py* can be run to solve the 300-20 batch identification problem while comparing with the combinatorial algorithm. In the meanwhile, you can change the 300-20 with 300-x by generating from *Batch access simulator.py*.
- The second one: We will train a model with generalization as provided within the folder *\\Trained model 300-x*. You need to modify the *Batch_Signatures_4actions.py* code to restore the previously commented out lines 227-255 and 293-295, and utilize *Main_300-x.py* for training and testing. For training and testing, you need to tweak the hyperparameters in *Main_300-x.py*, e.g. train_eps to 10000, etc. Then you may need to iterate the training on 60,000 eps, which requires constant tweaking and experimentation based on test results. The above process may take days and you may need a powerful computer. Similarly, to train the model at 2000-x, make changes to the code *Batch_Signatures_5actions.py* and *Main_2000-x.py*.

# An Experiment Showing Video

Just in case, the above description is unclear or too tedious, we have recorded a video of the code in action for reference. A new online (4K/2K) video of code version 2 address is https://youtu.be/FfwDH5HvooY
