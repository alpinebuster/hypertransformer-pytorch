# Call from the torch root:
#    `./scripts/omniglot_5shot_v1.sh` with flags
#    "--data_numpy_dir=<omniglot_cache> --train_log_dir=<output_path>"
# 
# e.g.
#    `nohup ./scripts/omniglot_5shot_v1.sh --data_dir=../omniglot --data_numpy_dir=../omniglot/cache --train_log_dir=../omniglot/logs/torch > omniglot_5shot.log 2>&1 &`
# 
./scripts/omniglot_1shot_v1.sh --samples_transformer=100 --samples_cnn=100 $@
