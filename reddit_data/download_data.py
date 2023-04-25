"""Download Reddit submissions given a list of submission ids."""
import csv
import re

import praw
import tqdm
from tqdm import tqdm


# based on: https://huggingface.co/datasets/sentence-transformers/reddit-title-body/blob/main/extraction_script/extract_title_selftext.py
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\[(.*)\]\(.*\)", "\g<1>", text)  # Markdown
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )  # URLs
    return text


# credentials must be specificed in praw.ini
# read-only mode is enough but an application has to be registered on Reddit: https://praw.readthedocs.io/en/stable/getting_started/authentication.html
reddit = praw.Reddit()

ids = open("submission_ids.txt", "r").read().splitlines()[1:]

submissions = []
for s in tqdm(reddit.info(fullnames=["t3_" + id_ for id_ in ids]), total=len(ids)):
    # ignore deleted or removed submissions (if submission or user is deleted)
    if s.selftext in ["[removed]", "[deleted]"]:
        continue

    title = clean_text(s.title)
    selftext = clean_text(s.selftext)
    submissions.append([s.id, title, selftext, s.subreddit])

with open("submissions.tsv", "w", encoding="utf8", newline="") as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
    tsv_writer.writerow(["id", "title", "selftext", "subreddit"])
    i = 0
    for row in submissions:
        i += 1
        tsv_writer.writerow(row)
