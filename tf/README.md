# Tensorflow implementation of (Continual) HyperTransformer

Tensorflow implementation of (Continual) HyperTransformer model as described in the
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

## Dataset NumPy Files

Datasets that are most widely used for model training (`omniglot`,
`miniimagenet` and `tieredimagenet`) are provided to the trainer as saved NumPy
arrays.

`omniglot` data can be downloaded
[from here](https://drive.google.com/file/d/1rmxwmCZwEJxuYGajYqFOaVUB7flmM4zI/view?usp=sharing).
Raw pickle files for `miniimagenet` can be downloaded
[here](https://www.kaggle.com/datasets/whitemoon/miniimagenet),
and `tieredimagenet` pickle files can be found in
[this repository](https://github.com/yinboc/few-shot-meta-baseline)
(courtesy of Kwonjoon Lee).

> `miniimagenet` and `tieredimagenet` cache files can be obtained using utilities
found in `utils`. 

## Examples

Folder `scripts` contains several training scripts for `omniglot`,
`miniimagenet` and `tieredimagenet` datasets.

They can be executed as follows:

```shell
./scripts/omniglot_1shot_v1.sh --data_numpy_dir=./data_numpy_dir --train_log_dir=./logs
tensorboard --logdir=../omniglot/logs
```

The evaluation script `evaluate_model.py` can be executed in parallel to
evaluate produced checkpoints while the model is being trained.
The evaluation script accepts the same parameters as the training script.

## Code Structure

The code consists of several parts:

1. _Task Generator_ — uses input dataset to generate _episodes_.
2. _Transformer IO (Input)_ — receives a support batch (typically a few samples
   per label) and encodes it producing sample embeddings that are passed to the
   Transformer model.
3. _Transformer_ — receives a set of input embeddings (encoding input samples)
   and produces a set of output embeddings (containing CNN model weights).
4. _Transformer IO (Output)_ — converts Transformer output to a list of tensors
   that contain unstructured CNN model weights.
5. _CNN Model Builder_ — uses generated weight tensors to create a CNN
   (uses TFv1 variable getter mechanism for final weight generation).

The model is trained end-to-end:
```txt
   ┌───────────────────────────┐
   │ (a) Generate a batch of   │
   │     samples for a new     │
   │         episode           │
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │ (b) Pass support samples  │
   │     to Transformer →      │
   │     Transformer generates │
   │           CNN             │
   └─────────────┬─────────────┘
                 │
                 ▼
   ┌───────────────────────────┐
   │ (c) Pass query samples    │
   │     through generated CNN │
   │     and compute loss →    │
   │     Backprop to train     │
   │       Transformer         │
   └───────────────────────────┘
```
