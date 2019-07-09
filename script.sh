allennlp train train_configs/defaults.jsonnet 
-s experiments/test1
--include-package dataset_readers.spider
--include-package models.semantic_parsing.spider_parser


allennlp predict experiments/gnn3 dataset/dev.json \
--predictor spider \
--use-dataset-reader \
--cuda-device=0 \
--output-file experiments/gnn3/prediction.sql \
--silent \
--include-package models.semantic_parsing.spider_parser \
--include-package dataset_readers.spider \
--include-package predictors.spider_predictor \
--weights-file experiments/gnn3/best.th