
import os

os.environ["HF_HUB_CACHE"] = r"E:\hug_cache\hub"
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
local_dir = r"E:\hug_cache\hub\models--bert-base-uncased--manual"

from transformers import AutoTokenizer
from bert_utils import load_and_split_data

TEXT_CSV = 'liar2_for_bert_text.csv'
META_CSV = 'liar2_for_bert_meta.csv'

df_text_train, df_meta_train, df_text_val, df_meta_val, df_text_test, df_meta_test = \
        load_and_split_data(TEXT_CSV, META_CSV,
                            test_size=0.20, val_size=0.50, random_state=200900)

tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
lengths = df_text_train['text_for_bert'].apply(lambda s: len(tok(s, truncation=False)['input_ids']))
print(lengths.describe())

print(lengths.describe(percentiles=[0.75, 0.90, 0.95, 0.99]))