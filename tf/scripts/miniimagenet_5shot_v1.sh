# Call `./scripts/miniimagenet_1shot_v1.sh` from the `tf` root with flags:
#   "--data_dir=<miniimagenet> --data_numpy_dir=<miniimagenet_cache> --train_log_dir=<output_path>"
# 
# e.g. `nohup ./scripts/miniimagenet_1shot_v1.sh --data_dir=../miniimagenet --data_numpy_dir=../miniimagenet/cache --train_log_dir=../miniimagenet/logs/tf > miniimagenet_5shot.log 2>&1 &`
# DS: `--train_dataset=miniimagenet` # or imagenette, emnist

./scripts/omniglot_1shot_v1.sh --samples_transformer=25 --samples_cnn=100 $@
