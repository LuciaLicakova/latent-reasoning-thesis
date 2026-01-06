# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Last update: Lucia Licakova, 2026-01-06

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_dataset(path, tokenizer, max_size=1000000000):
    # maximum sequence length (1024 for GPT-2)
    max_len = tokenizer.model_max_length
##    print(f"path: {path}")

    def tokenize_sample(sample):
        '''Example sample:
        {
            "question": "What is 2 + 2?",
            "steps": ["Add 2 and 2."],
            "answer": "4",
            "idx": 0
        }
        '''
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        
        total_len = (
            len(question_tokenized)
            + sum(len(s) for s in steps_tokenized)
            + len(answer_tokenized)
        )

        # in MATH dataset drop samples that exceed max length
        drop = ("math" in path and total_len > max_len)
        sample = {
            # List of token IDs
            "question_tokenized": [] if drop else question_tokenized,
            # List of lists of integers (token IDs), one list per step
            "steps_tokenized": [] if drop else steps_tokenized,
            # List of token IDs with EOS at the end
            "answer_tokenized": [] if drop else answer_tokenized,
            "idx": sample["idx"],
            "drop": drop,
        }
        return sample

    data = json.load(open(path))[:max_size]
    # Assign a unique identifier to each sample
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    # Create a HuggingFace dataset from the dictionary
    # where each key (question, steps, ...) maps to a list of values for that key across all samples
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    # If there are multiple GPUs available, handle distributed processing
    if dist.is_initialized() and torch.cuda.device_count() > 1:
        rank = dist.get_rank()
        if rank == 0:
            processed_dataset = dataset.map(
                tokenize_sample, remove_columns=list(dataset.features), num_proc=32
            )
            # drop too long samples
            processed_dataset = processed_dataset.filter(lambda x: not x["drop"])
            processed_dataset = processed_dataset.remove_columns(["drop"])
            # list for broadcast_object_list
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = dataset.map(
            # Remove the original fields (question, steps, answer) so only tokenized fields remain
            tokenize_sample, remove_columns=list(dataset.features), num_proc=4
        )
        # drop samples which exceed max length
        dataset = dataset.filter(lambda x: not x["drop"])
        dataset = dataset.remove_columns(["drop"])


    # Verify that tokenizing the question, steps, and answer separately matches
    # the result of tokenizing the whole text at once    
    if not "math" in path:
        # after dropping long samples the assertion no longer makes sense
        # data[0] is always the first raw JSON example
        # dataset[0] is the first example that survived filtering
        # the assertion is skipped for MATH dataset
        d = data[0]
        complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
        complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
            tokenizer.eos_token_id
        ]
        assert (
            complete_tokenized
            == dataset[0]["question_tokenized"]
            + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
            + dataset[0]["answer_tokenized"]
        )

    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    # -100 ensures the loss ignores these positions
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this to maximize the reuse of kv cache.
        The first latent token appears at the same position in all sequences.
        E.g.,
        
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx


        ("x" is word token, "-" is pad token)
        """

        earliest_latent = [
            # For each sample, find the index of the first latent token
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]
        # If there are continuous thoughts in the sequence
        if len(earliest_latent) > 0:
            # The rightmost first latent token among all sequences
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                # After padding all sequences have their first latent token aligned
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                # If a sample has no latent token, no padding is added
                else:
                    n_tok_pad = 0
                # Pads position_ids with zeros for the prepended padding tokens
                feature["position_ids"] = [0] * n_tok_pad + list(
                    # Remaining positions are sequential (0,1,2,...)
                    range(len(feature["input_ids"]))
                )
                # Prepend padding tokens (n_tok_pad) to the sequence (feature["input_ids"],
                # list of token IDs for one sequence in the batch)
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                # Pad the labels with -100 (label_pad_token_id) so the loss ignores these positions
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                # Pad attention mask with zeros so the model ignores padding tokens
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        # Pytorch
        return_tensors = "pt"
        # Determine label key based on the dataset
        label_name = "label" if "label" in features[0].keys() else "labels"
        # Extract only features used by the tokenizer (input_ids, attention_mask)
        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        # Use HuggingFace tokenizer to pad sequences without modifying labels or position_ids
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )
        # Extract labels from features
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        # Extract position IDs from features
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            # Pad with -100 (to ignore in loss) to match the maximum sequence length
            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            # Convert to tensor
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            # Pad position_ids with zeros to match the maximum sequence length
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            # Convert to tensor
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )
        # Dictionary of padded tensors input_ids, attention_mask, labels, position_ids
        return batch


def get_question_latent_dataset(
    # how many reasoning steps to abstract into latent tokens
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):

    def process_dataset(sample):
        # Determine the maximum number of latent tokens to include
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        # Don’t generate more latent tokens than there are reasoning steps
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )
        # The number of reasoning steps
        k = min(max_latent_stage, scheduled_stage)
        # Each language reasoning step is replaced by c_thought latent thoughts
        k *= configs.c_thought
        # Build the input token sequence for the model
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )
        # Dictionary for this processed sample
        return {
            # The full sequence of tokens (question + latent + markers)
            "input_ids": tokens,
            # The original index of the sample
            "idx": sample["idx"],
            # All tokens are attended to
            "attention_mask": [1] * len(tokens),
            # Tensor [0, 1, ...] representing the position of each token in the sequence
            "position_ids": list(range(len(tokens))),
        }
    # Apply process_dataset to all samples in the base dataset
    return base_dataset_valid.map(
        # The output only contains processed features
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=4
    )


def get_cot_latent_dataset(
    # How many reasoning steps to abstract into latent tokens
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):

    n_additional_tokens = 0 if no_special_marker else 2

    def process_dataset(sample):
        # With probability uniform_prob, choose a random reasoning stage
        if (
            random.random() < configs.uniform_prob
        ):
            scheduled_stage_to_train = random.choice(
                # all possible stages from 0 to the number of steps
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        # Otherwise use the fixed scheduled_stage
        else:
            scheduled_stage_to_train = scheduled_stage

            
        # We want more reasoning steps based on the stage than the model can handle
        if scheduled_stage_to_train > configs.max_latent_stage:
            # Skip all reasoning steps, replace them with latent tokens
            n_skip_steps = 10000
            # How many latent tokens to use for the replacement
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                # Use as many as there are steps (but capped at max_latent_stage)
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        # The stage is small enough, skip exactly scheduled_stage_to_train
        else:
            n_skip_steps, n_latent_tokens = (
                # How many real reasoning steps are hidden
                scheduled_stage_to_train,
                # How many latent tokens to insert in place of the skipped steps
                scheduled_stage_to_train,
            )
        # Skip all reasoning steps, don't replace them with latent tokens
        if configs.no_cot:
            n_skip_steps = 100
            n_latent_tokens = 0
            
        # Each reasoning step may correspond to multiple latent tokens
        n_latent_tokens *= configs.c_thought
        # Build token sequence
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            # Remaining reasoning steps (after skipping n_skip_steps)
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            # Ignore the question, latent tokens, start/end markers for loss computation
            "labels": [-100]
            * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            # Predict the actual reasoning steps and the final answer
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }
    # Process the entire dataset
    if dist.is_initialized() and torch.cuda.device_count() > 1:
        rank = dist.get_rank()
        if rank == 0:
            processed_dataset = base_dataset.map(
                process_dataset, remove_columns=list(base_dataset.features), num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        # Single GPU / CPU: save system resources with lower num_proc
        processed_dataset = base_dataset.map(
            process_dataset, remove_columns=list(base_dataset.features), num_proc=4
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset
