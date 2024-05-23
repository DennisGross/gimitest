# Gimitest: A Tool for Testing Reinforcement Learning Policies
This repository contains the experiments conducted in *Gimitest: A Tool for Testing Reinforcement Learning Policies*, which show the applicability of Gimitest.
Not only this implementation details how to reproduce the results of the paper, they can also be used as use cases of *Gimitest*.
Every use case is implemented in dedicated repositories.

## Installation

First, ensure that the library is correctly installed in the Python environment.
Then, install the few dependencies specific to the experiments with `pip install -r requirements.txt`.
The experiments rely on three [Gymnasium](https://gymnasium.farama.org/) environments, for which we train DQN and PPO agents with the library [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).
For reproducibility purpose, we provide the models used in the paper under `models/`.

## Failure Detection

We first demonstrate the use of the tool to test a policy solving [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
We reduce the search space by fixing the shape of the landscape and look for the initial forces applied to the agent that cause failures.
Therefore, the search space is $[-1000, 1000]^2$.
We adress the testing task with three methods:
 - Search-based testing, implemented as an evolutionary search.
 The search starts with a randomly generated population and each new population is formed by keeping the individuals that reveal the lowest rewards (i.e., weak run of the agent).
 - Fuzz-based testing, implemented as [MDPFuzz](https://sites.google.com/view/mdpfuzz/abstract) without coverage guidance (for simplicity).
 In a nutshell, the pool of inputs is maintained with mutations that decrease the reward accumulated by the agent (compared to the original input).
 As for the input selection, the later prioritizes the ones of high sensitivity, a measure that estimates the robustness of the agent against MDPFuzz's mutations.
 To that regard, the mutation operator is shared by the two frameworks.
 - A random testing baseline, implemented as a loop that iteratively generates a random input to test the agent.

### Testing the agent

Navigate to the folder *sbt* with `cd sbt`.
Run random testing with `python random_testing.py`.
By default, the results are logged in a file *random_testing.txt*.
Proceed similarly for with the scripts *es.py* and *fuzzing.py* to execute the two frameworks.
The default paths of the results are *es_testing.txt* and *fuzzing_logs.txt*.

## Adversarial Testing

We then illustrate the ability of Gimitest to access the agent by attacking the policy, i.e., finding states for which a small perturbation assumed (to be meaningless) lead to another decision.
Precisely, we implement the method proposed by [Huang et al. \[2017\]](https://arxiv.org/abs/1702.02284).
We consider the classical [Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) use case.
To run the demonstration, naviguate to `at/` and run `python fgsm_attacks.py`.

## Metamorphic Testing

We apply metamorphic testing with Gimitest.
We follow the approach proposed by [Eniser et al. \[2022\]](https://dl.acm.org/doi/abs/10.1145/3533767.3534392), where we look for *bugs* in the decision model.
*Bugs* are scenarios solved but the agent for which *simpler* versions however cause failures.
Finding those bugs require to define un/relaxation operations, that depend on the environment considered.
For this demonstration, we use the [Cart Pole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
We unrelax the states by increasing the absolute value of the pole' angle.

## TMARL Testing
TMARL (Turn-Based Multi-Agent Reinforcement Learning) testing is a specialized application of Gimitest, designed to evaluate the performance of agents in turn-based multi-agent environments. In this testing framework, we focus on the classic game of Connect Four. The objective is to assess the agent's capability to secure a win against a randomly behaving opponent. To achieve this, the agent is tested from various initial board states to determine its probability of winning.


## CMARL Testing
CMARL (Concurrent Multi-Agent Reinforcement Learning) testing represents another distinct application of Gimitest, aimed at evaluating agents in concurrent multi-agent environments. For this testing, we use the Waterworld environment, where agents operate simultaneously in a shared space. The evaluation involves testing the agents' performance with varying numbers of evaders (targets the agent needs to capture) and poisons (hazardous elements the agent must avoid). 

## Plotting the results

The results of any experiment can be plotted as in the paper with the scripts *plot_results.py*.
Remind to change the paths of the log files if needed.

