# GCI: A Graph Concept Interpretation Framework 

This repository contains an implementation of GCI. 
GCI is a (G)raph (C)oncept (I)nterpretation Framework, 
which can be used for _quantitatively_ verifying concepts extracted from GNNs, 
using provided human interpretations. 

Specifically, GCI encodes user-provided concept interpretations 
as functions, which can be used to quantitatively measure alignment between extracted 
concepts and user interpretations. 


## Abstract
Explainable AI (XAI) underwent a recent surge in research on concept extraction, 
focusing on extracting human-interpretable concepts from Deep Neural Networks. 
An important challenge facing concept extraction approaches is the difficulty of 
interpreting and evaluating discovered concepts, especially for complex tasks such as molecular property prediction. 
We address this challenge by presenting GCI: a (G)raph (C)oncept (I)nterpretation framework, used for quantitatively 
measuring alignment between concepts discovered from Graph Neural Networks (GNNs) and their corresponding human 
interpretations. GCI encodes concept interpretations as functions, which can be used to quantitatively measure 
the alignment between a given interpretation and concept definition. We demonstrate four applications of 
GCI: (i) quantitatively evaluating concept extractors, (ii) measuring alignment between concept extractors and 
human interpretations, (iii) measuring the completeness of interpretations with respect to an end task and (iv) 
a practical application of GCI to molecular property prediction, in which we demonstrate how to use chemical 
_functional groups_ to explain GNNs trained on molecular property prediction tasks, and implement 
interpretations with a 0.76 AUCROC completeness score.

![alt text](https://github.com/dmitrykazhdan/CME/blob/master/figures/vis_abs.png)


## Citing

If you find this code useful in your research, please consider citing:

```
@article{
TBC
}
```