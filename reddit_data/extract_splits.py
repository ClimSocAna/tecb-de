"""Script to generate splits for benchmarking text embedding clustering.
Based on Reddit data as retrieved by the official Reddit API."""
import random

import jsonlines
import pandas as pd

NUM_SPLITS = 10
MIN_LABELS = 10
MAX_LABELS = 50

random.seed(42)


def get_split(submissions, labels_mask, col_name):
    return (
        submissions[labels_mask]
        .sample(frac=1.0)
        .rename(columns={col_name: "sentences"})[["sentences", "labels"]]
        .to_dict("list")
    )


def write_sets(name, sets):
    with jsonlines.open(name, "w") as f_out:
        f_out.write_all(sets)


submissions = pd.read_csv("submissions.tsv", delimiter="\t")
submissions.head()

submissions = submissions.rename(
    columns={"title": "s2s", "selftext": "p2p", "subreddit": "labels"}
)
submissions["p2p"] = submissions["s2s"] + " " + submissions["p2p"]

subreddits = list(submissions["labels"].unique())
test_sets_s2s, test_sets_p2p = [], []
for _ in range(NUM_SPLITS):
    num_labels = random.randint(MIN_LABELS, MAX_LABELS)
    random.shuffle(subreddits)
    labels = subreddits[:num_labels]

    labels_mask = submissions.labels.isin(labels)
    test_sets_s2s.append(get_split(submissions, labels_mask, "s2s"))
    test_sets_p2p.append(get_split(submissions, labels_mask, "p2p"))

write_sets("s2s_test.jsonl", test_sets_s2s)
write_sets("p2p_test.jsonl", test_sets_p2p)
