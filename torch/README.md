# hypertransformer-pytorch
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

Folder `scripts` contains several training scripts for `omniglot`,
`miniimagenet` and `tieredimagenet` datasets.

They can be executed as follows:

```shell
./scripts/omniglot_1shot_v1.sh --data_numpy_dir=./data_numpy_dir --train_log_dir=./logs
```

The evaluation script `evaluate_model.py` can be executed in parallel to
evaluate produced checkpoints while the model is being trained.
The evaluation script accepts the same parameters as the training script.

> DDP (DistributedDataParallel) Supports

```sh
# `nproc_per_node` -> Number of GPUs
torchrun --nproc_per_node=2 ./hypertransformer/train.py \
  --num_layerwise_features=8 --default_num_channels=8 \
  --samples_transformer=20 --samples_cnn=60 --num_labels=20 --learning_rate=0.02 \
  --learning_rate_decay_steps=100000.0 --learning_rate_decay_rate=0.95 \
  --train_steps=4000000 --steps_between_saves=1000 --lw_key_query_dim=1.0 \
  --lw_value_dim=1.0 --lw_inner_dim=1.0 --cnn_model_name='maxpool-4-layer' \
  --embedding_dim=32 --num_layers=3 --stride=1 --heads=2 \
  --shared_feature_extractor='3-layer-bn' --shared_input_dim=1 --shared_features_dim=32 \
  --shared_feature_extractor_padding=same --layerwise_generator=joint \
  --nolw_use_nonlinear_feature --lw_weight_allocation=output --nolw_generate_bias \
  --nolw_generate_bn --nouse_decoder --noadd_trainable_weights --image_size=28 \
  --balanced_batches --per_label_augmentation --rotation_probability=0.0 \
  --boundary_probability=0.0 --smooth_probability=0.0 --contrast_probability=0.0 \
  --resize_probability=0.0 --negate_probability=0.0 --roll_probability=0.0 \
  --angle_range=30.0 --random_rotate_by_90 --train_dataset='omniglot:0-1149' \
  --test_dataset='omniglot:1200-1622' --eval_datasets='omniglot' \
  --num_task_evals=1024 --num_eval_batches=16 --eval_batch_size=100 \
  --shuffle_labels_seed=2022 --test_rotation_probability=0.0 \
  --test_smooth_probability=0.0 --test_contrast_probability=0.0 \
  --test_resize_probability=0.0 --test_negate_probability=0.0 \
  --test_roll_probability=0.0 --test_angle_range=-1.0
```

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
- (a) A batch of samples for a new episode is
generated; 
- (b) Support samples are passed to the Transformer, which generates a
CNN; 
- (c) Query samples are then passed through the generated CNN and the final CNN classification loss is used to train the Transformer.
