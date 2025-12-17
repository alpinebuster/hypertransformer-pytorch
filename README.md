# HyperTransformers

Pytorch implementation of (Continual) HyperTransformer model as described in the
[HyperTransformer](https://arxiv.org/abs/2201.04182) paper (_HyperTransformer:
Model Generation for Supervised and Semi-Supervised Few-Shot Learning_) and [Continual HyperTransformer](https://arxiv.org/abs/2301.04584) paper (_Continual HyperTransformer: A Meta-Learner for Continual Few-Shot Learning_).

This a transformer-based model for few-shot learning that generates weights of
a convolutional neural network (CNN) directly from support samples.
Since the dependence of a small generated CNN model on a specific task is
encoded by a high-capacity transformer model, the complexity of the large task
space is effectively decoupled from the complexity of individual tasks.
This method is particularly effective for small target CNN architectures where
learning a fixed universal task-independent embedding is not optimal and better
performance is attained when the information about the task can modulate all
model parameters. The implementation also supports a semi-supervised regime
utilizing unlabeled samples in the support set and further improving few-shot
performance.

## Examples

```sh
# cd ./tf/
cd ./torch/
poetry install --no-root

poetry env activate
# or
source $(poetry env info -p)/bin/activate
# or
. $(poetry env info -p)/bin/activate
```

## Code Structure

The source code consists of two main parts:

1. _tf_ — Tensorflow implementation.
2. _torch_ — Pytorch implementation.

The model is trained end-to-end: 
- (a) A batch of samples for a new episode is
generated; 
- (b) Support samples are passed to the Transformer, which generates a
CNN; 
- (c) Query samples are then passed through the generated CNN and the final CNN classification loss is used to train the Transformer.
