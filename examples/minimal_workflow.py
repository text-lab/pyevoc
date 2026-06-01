import pandas as pd

from pyevoc.data import DatasetConfig, standardise_dataset
from pyevoc.preprocessing import clean_corpus

raw = pd.DataFrame({
    "author": ["u1", "u2"],
    "id": ["d1", "d2"],
    "created": ["2024-01-01", "2024-01-02"],
    "forum": ["example", "example"],
    "body": ["Solar energy is important.", "Fossil fuel debates continue."],
})

config = DatasetConfig(
    column_map={"author": "user_id", "id": "doc_id", "created": "time", "forum": "source", "body": "text"},
    start_date="2024-01-01",
    end_date="2024-01-31",
)
corpus = standardise_dataset(raw, config)
corpus = clean_corpus(corpus)
print(corpus)
