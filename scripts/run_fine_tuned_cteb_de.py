import argparse
import csv
import logging
import os
import random
import types
from typing import Optional

import numpy as np
import torch
from ClusteringTasks import (
    AbsTaskFlexibleClustering,
    BlurbsClusteringP2P,
    BlurbsClusteringS2S,
    RedditClusteringP2P,
    RedditClusteringS2S,
    TenKGnadClusteringP2P,
    TenKGnadClusteringS2S,
)
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, datasets, losses
from sentence_transformers.util import batch_to_device
from tqdm.autonotebook import trange
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def write_results(
    task: AbsTaskFlexibleClustering,
    model,
    output_path: str,
    epoch: int,
    steps: int,
    v_measure: float,
):
    """Write evaluation results to file."""
    csv_headers = ["epoch", "steps", "v_measure"]
    csv_path = os.path.join(output_path, f"{task.description['name']}.json")
    if not os.path.isfile(csv_path):
        with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            writer.writerow([epoch, steps, v_measure])
    else:
        with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, steps, v_measure])


def eval_wrapper(tasks: list, output_path: Optional[str] = None):
    """Wrapper for TSDAE training procedure so custom mteb tasks can be evaluated."""

    def eval(model, output_path=output_path, epoch=0, steps=0):
        v_measures = []

        model.eval()
        with torch.no_grad():
            for task in tasks:
                v_measure = task.evaluate(model)["v_measure"]
                if output_path is not None:
                    write_results(task, model, output_path, epoch, steps, v_measure)
                v_measures.append(v_measure)

        return np.mean(v_measures)

    return eval


def encode_wrapper(tokenizer, batch_size: int = 32):
    """Wrapper to make instances of transformer models compatible with mteb (which uses encode method similar to SentenceTransformer models)."""

    def encode(self, sentences):
        embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for i in range(0, len(sentences), batch_size):
            encoded_input = encoded_input = tokenizer(
                sentences_sorted[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded_input = batch_to_device(encoded_input, self.device)
            output = self(**encoded_input)

            token_embeddings = output[0]
            attention_mask = encoded_input.attention_mask
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            embeddings.extend((sum_embeddings / sum_mask).detach().cpu())

        embeddings = [embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return np.asarray([emb.numpy() for emb in embeddings])

    return encode


class CustomTrainer(Trainer):
    """Subclassed Hugging Face Trainer so custom mteb tasks can be evaluated."""

    def __init__(self, tasks: list, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.performed_evals = 0
        self.model.encode = types.MethodType(
            encode_wrapper(
                self.data_collator.tokenizer, self.args.per_device_eval_batch_size
            ),
            self.model,
        )

    def evaluate(self, **kwargs):
        self.performed_evals += 1
        steps = self.performed_evals * self.args.eval_steps
        steps_per_epoch = len(self.train_dataset) / (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        epoch = int(np.floor((steps - 1) / steps_per_epoch))

        v_measures = []

        self.model.eval()
        with torch.no_grad():
            for task in self.tasks:
                v_measure = task.evaluate(self.model)["v_measure"]
                if self.args.output_dir is not None:
                    write_results(
                        task, self.model, self.args.output_dir, epoch, steps, v_measure
                    )
                v_measures.append(v_measure)

        return np.mean(v_measures)


class TokenizedSentencesDataset:
    """Wrapper for on-the-fly tokenization for MLM training."""

    def __init__(
        self,
        sentences: list[str],
        tokenizer,
        max_length: int,
        cache_tokenization: bool = False,
    ):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(
                self.sentences[item],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(
                self.sentences[item],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


model_names = [
    "deepset/gbert-base",
]

base_tasks = [
    BlurbsClusteringS2S,
    BlurbsClusteringP2P,
    TenKGnadClusteringS2S,
    TenKGnadClusteringP2P,
]
task_configs = [
    {
        "dim_red": None,
        "clustering_alg": "minibatch_kmeans",
        "clustering_params": {"random_state": 42},
    },
]

seed_list = [42, 1, 2]
use_fp16 = False

# MLM
mlm_prob = 0.15


def main(args: argparse.ArgumentParser):
    if args.include_reddit:
        base_tasks.extend([RedditClusteringS2S, RedditClusteringP2P])

    for model_name in model_names:
        for base_task in base_tasks:
            tasks = [base_task(**config) for config in task_configs]
            logger.info(tasks[0].description["name"])
            base_task_name = tasks[0].description["name"].split("{")[0]

            tasks[0].load_data()
            train_sentences = list(
                set(
                    [
                        sent
                        for split in tasks[0].dataset["test"]["sentences"]
                        for sent in split
                    ]
                )
            )

            # TSDAE
            train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

            for seed in seed_list:
                torch.manual_seed(seed)

                model = SentenceTransformer(model_name)

                result_folder = os.path.join(
                    "results",
                    model_name.split("/")[-1],
                    f"tsdae_ft_{str(seed)}",
                    base_task_name,
                )

                g = torch.Generator()
                g.manual_seed(seed)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=8,
                    shuffle=True,
                    drop_last=False,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                train_loss = losses.DenoisingAutoEncoderLoss(
                    model, decoder_name_or_path=model_name, tie_encoder_decoder=True
                )

                dev_evaluator = eval_wrapper(tasks, output_path=result_folder)

                logger.info(f"Start TSDAE training <<seed: {seed}>>")
                model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    evaluator=dev_evaluator,
                    evaluation_steps=512,
                    epochs=30,
                    weight_decay=0,
                    scheduler="constantlr",
                    optimizer_params={"lr": 3e-5},
                    output_path=result_folder,
                    save_best_model=True,
                    checkpoint_save_total_limit=1,
                    show_progress_bar=True,
                    use_amp=use_fp16,
                )

                logger.info("Training done")

            # MLM
            for seed in seed_list:
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                train_dataset = TokenizedSentencesDataset(
                    train_sentences, tokenizer, 512
                )  # handle max seq length
                data_collator = DataCollatorForWholeWordMask(
                    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
                )

                result_folder = os.path.join(
                    "results",
                    model_name.split("/")[-1],
                    f"mlm_ft_{str(seed)}",
                    base_task_name,
                )

                training_args = TrainingArguments(
                    output_dir=result_folder,
                    overwrite_output_dir=True,
                    num_train_epochs=30,
                    evaluation_strategy="steps",
                    per_device_train_batch_size=32,
                    per_device_eval_batch_size=128,
                    eval_steps=25,
                    save_steps=25,
                    logging_steps=25,
                    gradient_accumulation_steps=8,
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="v_measure",
                    prediction_loss_only=True,
                    fp16=use_fp16,
                    learning_rate=1e-04,
                    weight_decay=0.01,
                    lr_scheduler_type="constant_with_warmup",
                    warmup_ratio=0.06,
                    seed=seed,
                )

                trainer = CustomTrainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset="placeholder",
                    tasks=tasks,
                )

                logger.info(f"Start MLM training <<seed: {seed}>>")
                trainer.train()
                logger.info("Training done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-reddit", action="store_true")

    args = parser.parse_args()
    main(args)
