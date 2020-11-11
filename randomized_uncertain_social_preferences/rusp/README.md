# Randomized Uncertain Social Preferences
We share the environment code used in the work *Emergent Reciprocity and Team Formation from Randomized Uncertain Social Preferences* (TODO: ADD LINK).

The relevant code for randomized uncertain social preferences are in wrappers in *wrappers_rusp.py* --- Here we define a wrapper that defines a random reward sharing relationship graph per episode and transforms agents' reward accordingly. Each agent is given an independent uncertainty and noisy sample around this relationship graph. Tests for making sure observations get routed properly are in *test_wrapper_rusp.py*.

## Environments
 * *env_ipd.py*: 2 player infinite horizon prisoner's dilemma
 * *env_indirect_reciprocity.py*: n-player infinite horizon prisoner's dilemma where at each step 2 agents are randomly chosen to play
 * *env_prisoners_buddy.py*: an abstract game where agents must mutually choose each other and resist temptation to defect and change teams.
 * *env_oasis.py*: MUJOCO based survival game where the environment is resource constrained such that only a subset of agents can survive.