# Call from the tf root as
# `./scripts/miniimagenet_5shot_v1.sh` with flags
# "--data_numpy_dir=<miniimagenet_cache> --train_log_dir=<output_path>"
# e.g. `./scripts/miniimagenet_5shot_v1.sh --data_dir=../miniimagenet --data_numpy_dir=../miniimagenet/cache --train_log_dir=../miniimagenet/logs`

./scripts/miniimagenet_1shot_v1.sh --samples_transformer=25 --samples_cnn=100 $@
