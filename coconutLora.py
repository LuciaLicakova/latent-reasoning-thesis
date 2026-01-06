# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Last update: Lucia Licakova, 2025-01-06
# Adapted for Lora

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import GPTNeoForCausalLM
from peft import LoraConfig, get_peft_model
from transformers.cache_utils import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8
LATENT_WINDOW_SIZE = 3


class CoconutLearnableLora(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(CoconutLearnableLora, self).__init__()   
        self.gen_forward_cnt = 0
        # inject LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.base_causallm = get_peft_model(base_causallm, lora_config)
        self.base_causallm.print_trainable_parameters()
        # Freeze the base model parameters
        for name, param in self.base_causallm.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Safety assertion
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert trainable > 0, "No trainable LoRA parameters found"
        
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        # parameters for context
        self.latent_window_size = LATENT_WINDOW_SIZE
        # detect device from provided base model parameters
        device = next(self.base_causallm.parameters()).device
        # register the tensor as a learnable model parameter
        # create a tiny module to hold latent weights so we can wrap it with FSDP separately
        class _LatentModule(nn.Module):
            def __init__(self, size, device):
                super().__init__()
                self.latent_weights = nn.Parameter(torch.zeros(size, device=device))

        self.latent_module = _LatentModule(self.latent_window_size, device)
        # assert the sizes
        if self.latent_module.latent_weights.size(0) != self.latent_window_size:
            raise RuntimeError(
                f"latent window mismatch: latent_window_size={self.latent_window_size} "
                f"but latent_module.latent_weights.shape[0]={self.latent_module.latent_weights.size(0)}"
            )


        if isinstance(self.base_causallm, GPT2LMHeadModel):
            # GPT's architecture in Hugging Face is slightly different
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            # Convert token IDs (input_ids) into vectors (embeddings)
            # applicable to GPT-Neo, too
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []
        # Find all positions of latent tokens in the batch (indices of all non-zero elements of input)
        # Each row: (batch_idx, token_pos)
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        # A list of lists, one per batch item      
        # For each batch keeps only the positions of latent tokens
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        # The largest number of latent tokens in any example
        max_n_latents = max([len(l) for l in latent_lists])
        # Compute the whole sequence (from token 0 to the last token)
        next_compute_range = (0, input_ids.shape[1])
        # Convert token IDs into embeddings of shape (batch_size, seq_len, hidden_dim)
        inputs_embeds = self.embedding(input_ids)

        # If there are latent tokens in the batch, avoid computing embeddings for them before they are ready
        if max_n_latents > 0:
            # Only compute up to the earliest latent token across the batch
            next_compute_range = (0, latent_indices[:, 1].min().item())

        # Store past key/value pairs from the attention layers to
        # allow subsequent passes to reuse computations of tokens already processed
        # Cache instance: mutable, grows with each forward pass
        kv_cache = DynamicCache()

        
        # Iteratively replace latent tokens with hidden states from previous steps
        for pass_idx in range(max_n_latents):
            # On the first forward pass, past_key_values=kv_cache is empty
            # the model behaves as if no cache was passed
            # returned past_key_values is the same object, now populated
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0] : next_compute_range[1], :
                ],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            kv_cache = outputs.past_key_values
            # How many tokens were already in the cache before this forward pass
            # (prefix_length + newly_computed_length) - newly_computed_length
            # For the first pass, cache length = number of tokens just computed and hidden_states_offset = 0
            hidden_states_offset = kv_cache.get_seq_length() - outputs.hidden_states[-1].shape[1]

            logits.append(outputs.logits)

            next_compute_range = (
                # New start = previous end
                next_compute_range[1],
                (
                    # If this was the last latent token pass, jump to the end of the sequence
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    # Otherwise extend by one token (the next latent position)
                    else next_compute_range[1] + 1
                ),
            )
            # The final-layer hidden states will replace latent token embeddings in the next step
            hidden_states = outputs.hidden_states[-1]  

            # Feedback the continuous thoughts to the input_embeds

            # Decide which latent tokens get replaced with a computed vector during this pass
            # During each pass, we replace the pass_idx'th latent token in each sequence
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                # skip if there is no pass_idx'th latent token in this sequence
                if len(mask_list) > pass_idx
            ]

            # To avoid in-place operations on the big tensor inputs_embeds (batch_size, seq_len, hidden_size)
            # tensor_list contains one list per batch;
            # each of those inner lists contains one vector per position in the sequence
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Ensure weights are on the right device
            weights = self.latent_module.latent_weights.clone()
            # Replace each latent token position's embedding with
            # a weighted combination of the last n_tokens hidden states
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                local_token_idx = token_idx - hidden_states_offset
                # Determine how many previous tokens are available
                n_tokens = min(self.latent_window_size, local_token_idx, weights.size(0))
                if n_tokens <= 0:
                    continue

                hidden_slice = hidden_states[
                    batch_idx,
                    local_token_idx - n_tokens : local_token_idx,
                    :
                ]

                # how many weights we use (even if fewer tokens are available)
                # Select the last n_tokens entries WITHOUT creating a view
                raw = weights[-n_tokens:].clone()
                # normalise them so they sum up to 1
                w = torch.softmax(raw, dim=0)
                # Reshape the 1D weight vector to (n_tokens, 1), multiply the hidden states element-wise
                # Sum accross the n_tokens dimension, return a weighted combination of the previous hidden states
                # Use unsqueeze instead of view to avoid possible view issues
                weighted_hidden = (hidden_slice * w.unsqueeze(1)).sum(dim=0)
                
                tensor_list[batch_idx][token_idx] = weighted_hidden


            # Convert the Python lists back into a proper tensor of shape (batch, seq_len, hidden_size)
            inputs_embeds = torch.stack(
                [
                    # Now inputs_embeds contains replacements at the latent-token positions
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final pass: process the remaining tokens after all latent tokens have been replaced
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                # Keep all batches, pick only the unprocessed tokens, keep all hidden dimensions
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            # Only attend to tokens seen so far
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=kv_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        # for consistency
        kv_cache = outputs.past_key_values

        
        # Perform max_n_latents + 1 forward passes:
        # max_n_latents latent thoughts are scheduled in the current training stage;
        # compute a new latent thought with each pass and run an additional forward pass
        # to obtain a loss on the remaining text sequence
        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate logits from all passes to get predictions for each position in the sequence
        logits = torch.cat(logits, dim=-2)
        # Predicted token probabilities for every position except the last one (can’t predict beyond the sequence)
        shift_logits = logits[..., :-1, :].contiguous()
        # True token IDs for every position except the first one (can’t predict the very first token)
        shift_labels = labels[..., 1:].contiguous()
        # Each prediction is compared to the next token in the sequence

        
        loss_fct = CrossEntropyLoss()
        # Apply cross-entropy loss between predicted distributions and true labels
        loss = loss_fct(
            # Flatten for CrossEntropyLoss which expects
            # input shape (N, C) N examples, C classes, target shape (N,) class indices for each example
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # Output a single scalar loss for the entire batch
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        # Convert only the first sequence to a list of generated tokens without tracking gradients
        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder to match the forward method’s interface; not used.
        outputs = self.forward(
            input_ids,
            # attention_mask is all ones, all tokens are valid
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            # position_ids are sequential integers [0, 1, 2, ...]
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # Pick the token with the highest probability from the last position of the logits
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        # Generate the first token
        tokens.append(next_token)
        # Convert the chosen token to its embedding
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        # Append to the existing sequence embeddings
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Generate other tokens up to max_new_tokens
        for _ in range(max_new_tokens - 1):
            # Pass the current embeddings to the model
            outputs = self.base_causallm(
                inputs_embeds=new_inputs_embeds,
                use_cache=False,
            )
            self.gen_forward_cnt += 1
            # Choose the most probable token
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            # If the model predicts that the sequence should stop, no more tokens are generated
            if next_token == self.eos_token_id:
                break
            # Update the token list and embeddings for the next step
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # In FSDP all GPUs must perform the same number of forward passes
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # For the purpose of analysis
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
