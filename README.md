# EvoPolicy

This package solves reinforcement learning problems using evolutionary strategeies as outlined in [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

# Files

**.solver.py**

Solver contains the class EvoSolver which is used to interact with an environment following OpenAI's gym api. Interactions with the environment are used to train a neural network that models a policy. The EvoSolver can then be used to select an action based on a state.

**.network.py**

This contains the neural network which models the policy. You should not need to interact with the network directly, instead use the EvoSolver.

# Example

example_polecart.py contains example usage with the polecart environment.

# Requirements

Developed with pyton==3.6.9

Install requirements with:
> pip install -r requirements.txt