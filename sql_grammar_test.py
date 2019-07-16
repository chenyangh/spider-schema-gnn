import json
from dataset_readers.dataset_util.spider_utils import fix_number_value, disambiguate_items

data_path = 'dataset/train_spider.json'
tables_file = 'dataset/tables.json'

with open(data_path, "r") as data_file:

    json_obj = json.load(data_file)

    for total_cnt, ex in enumerate(json_obj):

        if 'query_toks' in ex:
            ex = fix_number_value(ex)

            try:
                query_tokens = disambiguate_items(ex['db_id'], ex['query_toks_no_value'],
                                                  tables_file, allow_aliases=False)
            except Exception as e:
                # there are two examples in the train set that are wrongly formatted, skip them
                print(f"error with {ex['query']}")
                print(e)