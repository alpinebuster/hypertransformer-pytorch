# Call from the root as
# `./hypertransformer/tf/scripts/omniglot_5shot_v1.sh` with flags
# "--data_numpy_dir=<omniglot_cache> --train_log_dir=<output_path>"
# e.g. `./hypertransformer/tf/scripts/omniglot_5shot_v1.sh --data_dir=./omniglot --data_numpy_dir=./omniglot/cache --train_log_dir=./omniglot/logs`

./hypertransformer/tf/scripts/omniglot_1shot_v1.sh --samples_transformer=100 --samples_cnn=100 $@
