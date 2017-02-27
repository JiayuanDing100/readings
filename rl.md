# Reinforcement Learning readings


### Overview
1. [DEEP REINFORCEMENT LEARNING: AN OVERVIEW](https://arxiv.org/pdf/1701.07274.pdf)<br/>
`--A good review to briefly summarize the latest work on deep RL.`


### Hierarchical RL
1. [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057)<br/>
`--A top-level value function (meta controller) learns a policy over subgoals, and a low-level function (controller) learns a policy over atomic actions to satisfy the given goal.`
2. [Stochastic Neural Networks for Hierarchical Reinforcement Learning](https://openreview.net/pdf?id=B1oK8aoxe)<br/>
`--First learn a span of skills in a pre-training environment, then train high-level policies over these skills in the downstream tasks.`
3. 

### Task Hierarchy
1. [Automatic Discovery and Transfer of MAXQ Hierarchies](http://engr.case.edu/ray_soumya/papers/maxq.icml08.pdf)<br/>
`--1). Discovery the subtask hierarchies by analyzing the casual and temporal relationships among the actions in the trajectory with DBN.`


### GAN with RL 
1. [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)<br/>
`--1). Sequence Generation, first GAN for text; 2). Adversarial Loss v.s. MLE`