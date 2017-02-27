# Dialogue readings


### Overview
1. []()<br/>


### Policy Transfer
1. [Dialogue policy learning for combinations of noise and user simulation: transfer results](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.6098&rep=rep1&type=pdf)<br/>
`--Compare policy transfer properties under different environment, and show that policy trained under high-noise conditions has better transfer properties.`
2. [Personalizing a Dialogue System with Transfer Learning](https://arxiv.org/abs/1610.02891)<br/>
`--Two Q networks, Q_g for common knowledge, Q_p for personalized networks; each user has a trained model; first train on source domain, then train on part of target domain; test with the rest of target domain.`
3. 

### Task-Completion Dialogue
1. [A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://arxiv.org/abs/1701.04024)<br/>
`--1). Seq2Seq (with Soft-attention for copying mechanism); 2). KB-Type encoding in Input; 3). For decoding, it does not say how to handle the (most) unseen kb-entities.`
2. [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/pdf/1605.07683.pdf)<br/>
`--1). Five tasks(template based); 2). Evaluation: Per-Dialogue accuracy, if the model can predict all the tasks (templates) correctly.`
3. []()<br/>


### Chit-chat
1. [Generative Deep Neural Networks for Dialogue: A Short Review](https://arxiv.org/abs/1611.06216)<br/>
`--1). Three encoder-decoder architectures; 2). Evaluation`
2. [Adversarial Evaluation of Dialogue Models](https://arxiv.org/abs/1701.08198)<br/>
`--1). Use adversarial loss as evaluation;`
3. [Adversarial Learning for Neural Dialogue Generation](https://arxiv.org/pdf/1701.06547.pdf)<br/>
`--1). Adversarial training for generation; 2). Adversarial evaluation`

### Image-Grounded Conversation:
1. [Image-Grounded Conversations: Multimodal Context for Natural Question and Response Generation](https://arxiv.org/pdf/1701.08251.pdf)<br/>
`--1). Question Generation & Response Generation; 2). 3-turn`


### Character-Level Response Generation
1. [Online Sequence-to-Sequence Active Learning for Open-Domain Dialogue Generation](https://arxiv.org/pdf/1612.03929.pdf)<br/>
`--1). Modified ByteNet for diversity-promoting; 2). Character-level response generation is working?(Not sure)`


