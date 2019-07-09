import json
import shutil
import sys

from allennlp.commands import main

config_file = "train_configs/defaults.jsonnet"

# Use overrides to train on CPU.
# overrides = json.dumps({"trainer": {"cuda_device": 0}})

# serialization_dir = "experiments/gnn3"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
# shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "experiments/gnn3",
    "dataset/dev.json",
    "--predictor", "spider",
    "--use-dataset-reader",
    "--cuda-device", '0',
    "--output-file", "experiments/gnn3/prediction.sql",
    "--silent",
    "--include-package", "dataset_readers.spider",
    "--include-package", "models.semantic_parsing.spider_parser",
    "--include-package", "predictors.spider_predictor",
    "--weights-file", "experiments/gnn3/best.th",
]

main()


# allennlp predict experiments/name_of_experiment dataset/dev.json \
# --predictor spider \
# --use-dataset-reader \
# --cuda-device=0 \
# --output-file experiments/name_of_experiment/prediction.sql \
# --silent \
# --include-package models.semantic_parsing.spider_parser \
# --include-package dataset_readers.spider \
# --include-package predictors.spider_predictor \
# --weights-file experiments/name_of_experiment/best.th