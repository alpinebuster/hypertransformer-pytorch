# Call from the tf root as
# `./scripts/omniglot_5shot_v1.sh` with flags
# "--data_numpy_dir=<omniglot_cache> --train_log_dir=<output_path>"
# e.g. `./scripts/omniglot_5shot_v1.sh --data_dir=../omniglot --data_numpy_dir=../omniglot/cache --train_log_dir=../omniglot/logs`

./scripts/omniglot_1shot_v1.sh --samples_transformer=100 --samples_cnn=100 $@
