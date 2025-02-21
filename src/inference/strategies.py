import warnings
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import math
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from utils import norm_logits


@dataclass
class DecoderOnlyDraftOutput(ModelOutput):
    """
    Base class for draft outputs of decoder-only generation models using speculative decoding.
    """

    sequences: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cand_probs: Optional[Tuple[torch.FloatTensor]] = None
    tree_att_mask: Optional[torch.FloatTensor] = None


@dataclass
class DecoderOnlyVerificationOutput(ModelOutput):
    """
    Base class for verification outputs of decoder-only generation models using speculative decoding.
    """

    sequences: torch.LongTensor = None
    target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    acceptance_count: Optional[int] = None


def _MCNS(
    ground_probs: torch.FloatTensor,
    cand_probs: Tuple[torch.FloatTensor],
    cand_tokens: torch.LongTensor,
) -> Optional[int]:
    ground_token = torch.multinomial(ground_probs, num_samples=1).item()

    for check_idx, cand_token in enumerate(cand_tokens):
        if ground_token == cand_token:
            return check_idx
    ground_probs[:] = 0
    ground_probs[ground_token] = 1
    return None


def _MCSSwoReplacement(
    ground_probs: torch.FloatTensor,
    cand_probs: Tuple[torch.FloatTensor],
    cand_tokens: torch.LongTensor,
) -> Optional[int]:
    cand_probs = cand_probs.to(ground_probs.device)
    for check_idx, cand_token in enumerate(cand_tokens):
        accept_threshold = ground_probs[cand_token] / cand_probs[cand_token]
        if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
            return check_idx
        else:
            ground_probs -= cand_probs
            ground_probs = torch.nn.functional.relu(ground_probs, inplace=True)
            ground_probs /= ground_probs.sum()
            cand_probs[cand_token] = 0
            cand_probs = cand_probs / cand_probs.sum()
    return None


def _MCSSwReplacement(
    ground_probs: torch.FloatTensor,
    cand_probs: Tuple[torch.FloatTensor],
    cand_tokens: torch.LongTensor,
) -> Optional[int]:
    cand_probs = cand_probs.to(ground_probs.device)
    for check_idx, cand_token in enumerate(cand_tokens):
        accept_threshold = ground_probs[cand_token] / cand_probs[cand_token]
        if torch.rand(1, device=accept_threshold.device) <= accept_threshold:
            return check_idx
        else:
            ground_probs -= cand_probs
            ground_probs = torch.nn.functional.relu(ground_probs, inplace=True)
            ground_probs /= ground_probs.sum()
    return None


class Strategy:
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int],
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        self.k_config = k_config
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_model_device = draft_model.model.get_input_embeddings().weight.device
        self.target_model_device = (
            target_model.model.get_input_embeddings().weight.device
        )
        self.max_draft_len = len(k_config)
        self.draft_model_temp = draft_model_temp
        self.target_model_temp = target_model_temp
        self.replacement = replacement
        self.speculative_sampling = speculative_sampling
        self.top_k = top_k
        self.top_p = top_p

        self.acceptance_check: Callable[
            [torch.FloatTensor, Tuple[torch.FloatTensor], torch.LongTensor],
            Optional[int],
        ] = None
        if speculative_sampling:
            if replacement:
                self.acceptance_check = _MCSSwReplacement
                if draft_model_temp == 0:
                    warnings.warn(
                        (
                            "You have set Temp=0 and are using sampling with replacement. "
                            "As a result, all the candidates obtained are the same, causing "
                            "the MCSD algorithm to degenerate into the vanilla SD."
                        ),
                        category=UserWarning,
                        stacklevel=3,
                    )
            else:
                self.acceptance_check = _MCSSwoReplacement
                if self.top_k > 0 or self.top_p > 0:
                    warnings.warn(
                        (
                            "You have set replacement=False and are using top-k or top-p sampling. "
                            "As a result, it might cause unexpected error if the number of non-zero elements"
                            "in sampling distribution is smaller than the number of elements to sample."
                        ),
                        category=UserWarning,
                        stacklevel=3,
                    )

        else:
            if replacement:
                warnings.warn(
                    (
                        "`replacement` is not applicable when `speculative_sampling` is False."
                        "The acceptance check algorithm defaults to MCNS (Multi-Candidate Naive Sampling)"
                        " when `speculative_sampling=False`."
                    ),
                    category=UserWarning,
                    stacklevel=3,
                )
            self.acceptance_check = _MCNS

    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        raise NotImplementedError

    def acceptance_check(self, ground_probs, cand_probs, cand_tokens) -> Optional[int]:
        raise NotImplementedError

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        cand_probs: Optional[Tuple[torch.FloatTensor]],
    ) -> DecoderOnlyVerificationOutput:
        raise NotImplementedError


class BatchStrategy(Strategy):
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int],
        draft_model_temp=1,
        target_model_temp=1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k = 10,
        top_p = 0.9,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            k_config,
            draft_model_temp,
            target_model_temp,
            replacement,
            speculative_sampling,
            top_k,
            top_p,
        )

        reversed_prod_size = [1]
        for i in range(1, self.max_draft_len):
            reversed_prod_size.insert(0, reversed_prod_size[0] * k_config[-i])

        self.reversed_prod_size = reversed_prod_size

    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = []
        for step in range(self.max_draft_len):
            step_k = self.k_config[step]
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids
            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            hidden_states = outputs.last_hidden_state

            logits = self.draft_model.lm_head(hidden_states[:, -1])

            past_key_values = list(outputs.past_key_values)

            if self.draft_model_temp == 0:
                if not self.replacement:
                    topk_logit, topk_index = logits.topk(k=step_k, dim=-1)  # batch x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                    cand_tokens = topk_index.view(-1, 1)
                else:
                    topk_logit, topk_index = logits.topk(k=1, dim=-1)  # batch x k
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, value=1)
                    cand_tokens = topk_index.view(-1, 1)
                    cand_tokens = torch.repeat_interleave(cand_tokens, step_k, dim=0)
            else:
                step_cand_probs = norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p)
                cand_tokens = torch.multinomial(
                    step_cand_probs,
                    step_k,
                    replacement=self.replacement,
                ).view(-1, 1)

            cand_probs.append(step_cand_probs)

            input_ids = input_ids.repeat_interleave(step_k, dim=0)
            input_ids = torch.cat(
                (
                    input_ids,
                    cand_tokens,
                ),
                dim=1,
            )
            if step + 1 != self.max_draft_len:
                for i in range(len(past_key_values)):
                    past_key_values[i] = (
                        past_key_values[i][0].repeat_interleave(step_k, dim=0),
                        past_key_values[i][1].repeat_interleave(step_k, dim=0),
                    )

        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
        )

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        cand_probs: Optional[Tuple[torch.FloatTensor]],
    ) -> DecoderOnlyVerificationOutput:
        input_ids = input_ids.to(self.target_model_device)
        batch_size, input_len = input_ids.size()
        if target_model_past_key_values is not None:
            pruned_input_ids = input_ids[
                :, target_model_past_key_values[0][0].size(2) :
            ]
            for i in range(len(target_model_past_key_values)):
                target_model_past_key_values[i] = (
                    target_model_past_key_values[i][0].repeat_interleave(
                        batch_size, dim=0
                    ),
                    target_model_past_key_values[i][1].repeat_interleave(
                        batch_size, dim=0
                    ),
                )
        else:
            pruned_input_ids = input_ids

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=target_model_past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        target_model_past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(hidden_states[:, -self.max_draft_len - 1 :])

        if self.target_model_temp == 0:
            _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
            ground_probs = torch.zeros_like(logits)
            ground_probs.scatter_(dim=2, index=topk_index, value=1)
        else:
            ground_probs = norm_logits(logits, self.target_model_temp, self.top_k, self.top_p)


        unverified_input_ids = input_ids[:, -self.max_draft_len :]

        assert ground_probs.size(1) == unverified_input_ids.size(1) + 1

        cand_probs_idx = 0
        alive_group_id = 0

        for depth in range(self.max_draft_len):
            verify_batch_ids = [
                alive_group_id + group_offset * self.reversed_prod_size[depth]
                for group_offset in range(self.k_config[depth])
            ]
            accept_idx_bias = self.acceptance_check(
                ground_probs[alive_group_id, depth],
                cand_probs[depth][cand_probs_idx],
                unverified_input_ids[verify_batch_ids, depth],
            )
            if accept_idx_bias is not None:
                alive_group_id = verify_batch_ids[accept_idx_bias]
                cand_probs_idx = accept_idx_bias + cand_probs_idx * self.k_config[depth]
                if depth == self.max_draft_len - 1:
                    depth = self.max_draft_len
            else:
                break
        input_ids = input_ids[alive_group_id, : input_len - self.max_draft_len + depth]
        endpoint_token = torch.multinomial(
            ground_probs[alive_group_id, depth], num_samples=1
        ).to(device=input_ids.device)

        input_ids = torch.cat((input_ids, endpoint_token))

        input_ids.unsqueeze_(0)

        for i in range(len(target_model_past_key_values)):
            target_model_past_key_values[i] = (
                target_model_past_key_values[i][0][
                    None, alive_group_id, :, : input_len - self.max_draft_len + depth
                ],
                target_model_past_key_values[i][1][
                    None, alive_group_id, :, : input_len - self.max_draft_len + depth
                ],
            )
        for i in range(len(draft_model_past_key_values)):
            draft_model_past_key_values[i] = (
                draft_model_past_key_values[i][0][
                    None,
                    alive_group_id // self.k_config[-1],
                    :,
                    : input_len - self.max_draft_len + depth,
                ],
                draft_model_past_key_values[i][1][
                    None,
                    alive_group_id // self.k_config[-1],
                    :,
                    : input_len - self.max_draft_len + depth,
                ],
            )
        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=depth,
        )


def get_tree_attn_self_mask(k_config: Tuple[int]):
    k_config = torch.tensor(k_config, dtype=torch.int)
    prod_size = torch.cumprod(k_config, dim=0)
    mask_size = prod_size.sum().item()
    attn_mask = torch.zeros((mask_size, mask_size), dtype=torch.bool)
    attn_mask = attn_mask.diagonal_scatter(torch.ones(mask_size))
    # run BFS
    idx_queue = [
        (0, None, idx) for idx in list(range(k_config[0]))
    ]  # each node: (depth, parent, idx)
    while len(idx_queue) != 0:
        depth, parent, idx = idx_queue.pop(0)
        if parent is not None:
            attn_mask[idx, : parent + 1] = attn_mask[parent, : parent + 1]

        if depth != len(k_config) - 1:
            idx_base = prod_size[:depth].sum().item()
            child_idx_base = prod_size[: depth + 1].sum().item()
            for child_idx_bias in range(k_config[depth + 1]):
                real_child_idx = (
                    (idx - idx_base) * k_config[depth + 1]
                    + child_idx_base
                    + child_idx_bias
                )
                idx_queue.append((depth + 1, idx, real_child_idx))
    return attn_mask


class TreeStrategy(Strategy):
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int],
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            k_config,
            draft_model_temp,
            target_model_temp,
            replacement,
            speculative_sampling,
            top_k,
            top_p,
        )

        prod_size = torch.cumprod(torch.tensor(k_config, dtype=torch.int), dim=0)
        prod_size = torch.cat((torch.zeros(1).to(prod_size), prod_size)).tolist()
        self.prod_size = prod_size
        self.cumulative_prod_size = torch.cumsum(
            torch.tensor(prod_size), dim=0
        ).tolist()

        self.tree_attn_self_mask = get_tree_attn_self_mask(k_config).to(
            device=self.draft_model_device
        )

    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = []
        step_tree_attn_mask = None
        position_ids = None
        init_input_length = input_ids.size(1)
        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = input_ids
        for step in range(self.max_draft_len):
            step_k = self.k_config[step]

            # prepare attn mask
            if step != 0:
                step_tree_attn_self_mask = self.tree_attn_self_mask[
                    self.cumulative_prod_size[step - 1] : self.cumulative_prod_size[
                        step
                    ],
                    : self.cumulative_prod_size[step],
                ]
                position_ids = torch.full(
                    (1, self.prod_size[step]),
                    init_input_length + step - 1,
                    dtype=torch.long,
                    device=self.draft_model_device,
                )
                context_attn_mask = torch.ones(
                    (self.prod_size[step], init_input_length), dtype=torch.bool
                ).to(self.tree_attn_self_mask)
                step_tree_attn_mask = torch.cat(
                    (context_attn_mask, step_tree_attn_self_mask), dim=1
                )

            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                tree_attn_mask=step_tree_attn_mask,
                position_ids=position_ids,
            )

            hidden_states = outputs.last_hidden_state

            if step == 0:
                hidden_states = hidden_states[0, -1:]
            else:
                hidden_states = hidden_states[0]
            logits = self.draft_model.lm_head(hidden_states)  # seq_len x hidden_dim

            past_key_values = list(outputs.past_key_values)

            if self.draft_model_temp == 0:
                if not self.replacement:
                    topk_logit, topk_index = logits.topk(
                        k=step_k, dim=-1
                    )  # seq_len x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                    cand_tokens = topk_index.view(1, -1)
                else:
                    topk_logit, topk_index = logits.topk(k=1, dim=-1)  # seq_len x k
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, value=1)
                    cand_tokens = topk_index.view(1, -1)
                    cand_tokens = torch.repeat_interleave(cand_tokens, step_k, dim=1)
            else:
                step_cand_probs = norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p)

                cand_tokens = torch.multinomial(
                    step_cand_probs, step_k, replacement=self.replacement
                ).view(1, -1)
            cand_probs.append(step_cand_probs)

            pruned_input_ids = cand_tokens

            input_ids = torch.cat((input_ids, pruned_input_ids), dim=1)

        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
        )

    def _forward_target_model(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ):
        input_ids = input_ids.to(self.target_model_device)
        tree_attn_len = self.tree_attn_self_mask.size(0)
        init_input_length = input_ids.size(1) - tree_attn_len
        init_forward = False

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = input_ids
            init_forward = True

        if init_forward:
            tree_attn_mask = torch.zeros(
                (input_ids.size(1), input_ids.size(1)),
                dtype=torch.bool,
                device=self.target_model_device,
            )
            mask_cond = torch.arange(
                tree_attn_mask.size(-1), device=self.target_model_device
            )
            tree_attn_mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(tree_attn_mask.size(-1), 1), 1
            )
            tree_attn_mask[-tree_attn_len:, -tree_attn_len:] = self.tree_attn_self_mask
            position_ids = tree_attn_mask.sum(dim=1) - 1

        else:
            tree_attn_mask = torch.ones(
                (
                    tree_attn_len + 1,
                    input_ids.size(1),
                ),  # there is one token not stored in the kv values
                dtype=torch.bool,
                device=self.target_model_device,
            )

            tree_attn_mask[1:, init_input_length:] = self.tree_attn_self_mask
            tree_attn_mask[0, init_input_length:] = 0
            position_ids = tree_attn_mask.sum(dim=1) - 1

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(
            hidden_states[:, -tree_attn_len - 1 :]
        )  # 1 x seq_len x hidden_dim
        return logits, past_key_values

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        cand_probs: Optional[Tuple[torch.FloatTensor]],
    ) -> DecoderOnlyVerificationOutput:
        input_ids = input_ids.to(self.target_model_device)
        logits, target_model_past_key_values = self._forward_target_model(
            input_ids, target_model_past_key_values
        )
        logits = logits[0]  # seq_len x hidden_dim
        tree_attn_len = self.tree_attn_self_mask.size(0)
        unverified_tokens = input_ids[0, -tree_attn_len:]
        init_input_length = input_ids.size(1) - tree_attn_len

        if self.target_model_temp == 0:
            _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
            ground_probs = torch.zeros_like(logits)
            ground_probs.scatter_(dim=1, index=topk_index, value=1)
        else:
            ground_probs = norm_logits(logits, self.target_model_temp, self.top_k, self.top_p)

        current_ground_prob = ground_probs[0]
        ground_probs = ground_probs[1:]

        keep_indices = list(range(init_input_length))
        to_drop_len = 0
        idx_group_bias = 0
        cand_probs_idx = 0

        for depth in range(self.max_draft_len):
            idx_base = self.cumulative_prod_size[depth] + idx_group_bias
            accept_idx_bias = self.acceptance_check(
                current_ground_prob,
                cand_probs[depth][cand_probs_idx],
                unverified_tokens[idx_base : idx_base + self.k_config[depth]],
            )
            if accept_idx_bias is not None:
                global_idx = idx_base + accept_idx_bias
                current_ground_prob = ground_probs[global_idx]
                keep_indices.append(init_input_length + global_idx)
                if depth == self.max_draft_len - 1:
                    to_drop_len += 1
                    depth = self.max_draft_len
                else:
                    cand_probs_idx = idx_group_bias + accept_idx_bias
                    idx_group_bias = cand_probs_idx * self.k_config[depth + 1]
            else:
                break

        keep_indices = torch.tensor(
            keep_indices, dtype=torch.long, device=self.target_model_device
        )
        if to_drop_len != 0:
            draft_keep_indices = keep_indices[: len(keep_indices) - to_drop_len]
        else:
            draft_keep_indices = keep_indices

        tail_ground_token = torch.multinomial(current_ground_prob, num_samples=1).to(
            device=input_ids.device
        )

        input_ids = input_ids.index_select(dim=1, index=keep_indices)
        input_ids = torch.cat((input_ids, tail_ground_token[None]), dim=1)

        for i in range(len(target_model_past_key_values)):
            keep_indices = keep_indices.to(
                device=target_model_past_key_values[i][0].device
            )
            target_model_past_key_values[i] = (
                target_model_past_key_values[i][0].index_select(
                    dim=2, index=keep_indices
                ),
                target_model_past_key_values[i][1].index_select(
                    dim=2, index=keep_indices
                ),
            )
        for i in range(len(draft_model_past_key_values)):
            draft_model_past_key_values[i] = (
                draft_model_past_key_values[i][0].index_select(
                    dim=2, index=draft_keep_indices
                ),
                draft_model_past_key_values[i][1].index_select(
                    dim=2, index=draft_keep_indices
                ),
            )

        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=depth,
        )



class BatchMTADStrategy(Strategy):
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int],
        beam_width: int,
        accept_thres: float=0.5,
        draft_model_temp=1,
        target_model_temp=1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k = 10,
        top_p = 0.9,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            k_config,
            draft_model_temp,
            target_model_temp,
            replacement,
            speculative_sampling,
            top_k,
            top_p,
        )
        self.beam_width = beam_width
        self.log_accept_thres = math.log(accept_thres)
        
        if target_model_temp == 0:
            warnings.warn(
                        (
                            "For MTAD, the target model temperature shouldn't be 0, there is no performance improvement"
                        ),
                        category=UserWarning,
                        stacklevel=3,
                    )


    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        """ use beam sampling strategy to generate a single draft sequence """
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = None
        log_beam_prob = None
        #token_dist_hist = None
        for step in range(self.max_draft_len):
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids

            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            hidden_states = outputs.last_hidden_state

            logits = self.draft_model.lm_head(hidden_states[:, -1])

            past_key_values = list(outputs.past_key_values)

            if log_beam_prob is None: # this is the first step of beam sampling
                # compute beam probability
                # do sampling
                if self.draft_model_temp > 0:
                    step_cand_probs = norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p) # 1 * vocab_size

                    cand_tokens = torch.multinomial(
                        step_cand_probs,
                        self.beam_width,
                        replacement = self.replacement,
                        ).view(-1,1)
                else:
                    topk_logit, topk_index = logits.topk(
                        k=self.beam_width, dim=-1
                    )  # seq_len x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                    cand_tokens = topk_index.view(-1, 1)

                log_beam_prob = step_cand_probs[:,cand_tokens].view(self.beam_width,1).log_softmax(dim=0)
                cand_probs = step_cand_probs[:,cand_tokens].view(-1,1).log()

                #token_dist_hist = step_cand_probs.repeat_interleave(self.beam_width, dim=0)[:,None,:]
                # modify input_ids based on sampling results
                input_ids = input_ids.repeat_interleave(self.beam_width, dim=0)
                input_ids = torch.cat(
                  (
                    input_ids,
                    cand_tokens,
                  ),
                  dim=1,
                )

                # modify key value cache 
                if step + 1 != self.max_draft_len:
                    for i in range(len(past_key_values)):
                        past_key_values[i] = (
                            past_key_values[i][0].repeat_interleave(self.beam_width, dim=0),
                            past_key_values[i][1].repeat_interleave(self.beam_width, dim=0),
                        )

                 

            else:
                """ log_beam_prob shape should be (num_beams, 1) """
                """ logits shape should be (num_beams, num_vocab) """
                # do sampling
                if self.draft_model_temp > 0:
                    # compute beam probability
                    step_cand_probs = torch.log(norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p))

                    vocab_size = step_cand_probs.size(-1)
                    beam_probs = (log_beam_prob + step_cand_probs).view(-1).exp()

                    cand_beams = torch.multinomial(
                        beam_probs,
                        self.beam_width,
                        replacement = self.replacement,
                        ).view(-1,1)
                else:
                    step_cand_probs = torch.log_softmax(logits, dim=-1)
                    vocab_size = step_cand_probs.size(-1)
                    beam_probs = (log_beam_prob + step_cand_probs).view(-1).exp()

                    topk_logit, topk_index = beam_probs.topk(
                        k=self.beam_width, dim=-1
                    )  # seq_len x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    beam_probs = torch.zeros_like(beam_probs)
                    beam_probs.scatter_(dim=0, index=topk_index, src=topk_probs)
                    cand_beams = topk_index.view(-1, 1)



                log_beam_prob = beam_probs[cand_beams].log_softmax(dim=0) # shape should be (beam_width, 1)

                # modify input_ids based on sampling results
                beam_idx = torch.div(cand_beams, vocab_size, rounding_mode = 'floor').long().view(-1)
                tokens = cand_beams % vocab_size


                log_token_probs = step_cand_probs.view(-1)
                log_sampled_token_probs = log_token_probs[cand_beams] # shape (beam_width, 1)
                cand_probs = torch.cat((cand_probs[beam_idx], log_sampled_token_probs), dim=1)

                #token_dist_hist = token_dist_hist[beam_idx]
                #cur_token_dist = step_cand_probs.exp()[beam_idx][:, None,:]
                #token_dist_hist = torch.cat((token_dist_hist, cur_token_dist), dim=1)

                if step + 1 == self.max_draft_len:
                    final_beam_idx = torch.argmax(log_beam_prob)
                    beam_idx = beam_idx[final_beam_idx]
                    tokens = tokens[final_beam_idx].view(1,1)
                    cand_probs = cand_probs[final_beam_idx]

                    input_ids = input_ids[beam_idx][None, :]
                    #token_dist_hist = token_dist_hist[final_beam_idx]
                else:
                    input_ids = input_ids[beam_idx]

                input_ids = torch.cat(
                        (
                            input_ids,
                            tokens,
                        ),
                        dim=1,
                )

                # modify key value cache
                for i in range(len(past_key_values)):
                    if step + 1 < self.max_draft_len:
                        past_key_values[i] = (
                            past_key_values[i][0][beam_idx],
                            past_key_values[i][1][beam_idx],
                            )
                    else:
                        past_key_values[i] = (
                            past_key_values[i][0][beam_idx:beam_idx+1],
                            past_key_values[i][1][beam_idx:beam_idx+1],
                            )

        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
        )

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        cand_probs: Optional[Tuple[torch.FloatTensor]],
    ) -> DecoderOnlyVerificationOutput:
        input_ids = input_ids.to(self.target_model_device)
        batch_size, input_len = input_ids.size()
        assert batch_size == 1
        if target_model_past_key_values is not None:
            pruned_input_ids = input_ids[
                :, target_model_past_key_values[0][0].size(2) :
            ]
        else:
            pruned_input_ids = input_ids

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=target_model_past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        target_model_past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(hidden_states[:, -self.max_draft_len - 1 :])

        if self.target_model_temp == 0:
            _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
            ground_probs = torch.zeros_like(logits)
            ground_probs.scatter_(dim=2, index=topk_index, value=1)
        else:
            ground_probs = norm_logits(logits, self.target_model_temp, self.top_k, self.top_p)

        unverified_input_ids = input_ids[:, -self.max_draft_len :]

        assert ground_probs.size(1) == unverified_input_ids.size(1) + 1

        cand_probs_idx = 0
        alive_group_id = 0

        log_cum_target_prob = 0
        log_cum_draft_prob = 0
        acc_len = 0

#        cand_probs, token_dist_hist = cand_probs
#        cand_probs = tuple(cand_probs)

        for depth in range(self.max_draft_len):
            log_cum_draft_prob += cand_probs[depth]
            token = unverified_input_ids[0, depth]
            log_cum_target_prob += ground_probs[0, depth, token].log()


            if log_cum_target_prob - log_cum_draft_prob > self.log_accept_thres:
                acc_len = depth + 1
            else:
                pass

        input_ids = input_ids[:, : input_len - self.max_draft_len + acc_len]
        endpoint_token = torch.multinomial(
                ground_probs[:, acc_len], num_samples=1
                ).to(device=input_ids.device)

        input_ids = torch.cat((input_ids, endpoint_token), dim=-1)
        for i in range(len(target_model_past_key_values)):
            target_model_past_key_values[i] = (
                    target_model_past_key_values[i][0][:, :, :input_len - self.max_draft_len + acc_len],
                    target_model_past_key_values[i][1][:, :, :input_len - self.max_draft_len + acc_len],
                    )
        for i in range(len(draft_model_past_key_values)):
            draft_model_past_key_values[i] = (
                    draft_model_past_key_values[i][0][:, :, :input_len - self.max_draft_len + acc_len],
                    draft_model_past_key_values[i][1][:, :, :input_len - self.max_draft_len + acc_len],
                    )


        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=acc_len,
        )

class TreeMTADStrategy(Strategy):
    def __init__(
        self,
        draft_model,
        target_model,
        k_config: Tuple[int],
        beam_width: int,
        accept_thres: float=0.5,
        draft_model_temp=1,
        target_model_temp=1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k = 10,
        top_p = 0.9,
    ) -> None:
        super().__init__(
            draft_model,
            target_model,
            k_config,
            draft_model_temp,
            target_model_temp,
            replacement,
            speculative_sampling,
            top_k,
            top_p,
        )
        self.beam_width = beam_width
        self.log_accept_thres = math.log(accept_thres)
        self.num_token_per_iter = beam_width * self.max_draft_len
        
        if target_model_temp == 0:
            warnings.warn(
                        (
                            "For MTAD, the target model temperature shouldn't be 0, there is no performance improvement"
                        ),
                        category=UserWarning,
                        stacklevel=3,
                    )


    def generate_draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ) -> DecoderOnlyDraftOutput:
        """ use beam sampling strategy to generate a single draft sequence """
        input_ids = input_ids.to(self.draft_model_device)
        cand_probs = None
        log_beam_prob = None
        #token_dist_hist = None
        input_len = input_ids.size(1)
        tree_att_mask = torch.full(
                                    (
                                      self.num_token_per_iter, 
                                      self.num_token_per_iter + input_len,
                                    ),
                                    True,
                                  )
        tree_att_mask[:, input_len:] = False
        step_tree_att_mask = None
        position_ids = None
        self.beam_idx_history = []
        self.draft_p_history = []

        for step in range(self.max_draft_len):
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids

            bias = self.beam_width * step
            if step > 0:
                # prepare special attention mask and position_ids
                step_tree_att_mask = tree_att_mask[
                                                     bias-self.beam_width:bias,
                                                     :input_len + bias
                                                  ].to(self.draft_model_device)
                position_ids = torch.full(
                                           (
                                             1, self.beam_width,
                                           ),
                    input_len + step - 1,
                    dtype=torch.long,
                    device=self.draft_model_device,
                )
                #print(step_tree_att_mask[:,-10:])


            outputs: BaseModelOutputWithPast = self.draft_model.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                tree_attn_mask=step_tree_att_mask,
                position_ids=position_ids,
            )

            hidden_states = outputs.last_hidden_state


            past_key_values = list(outputs.past_key_values)

            if log_beam_prob is None: # this is the first step of beam sampling
                logits = self.draft_model.lm_head(hidden_states[:, -1])
                # compute beam probability
                # do sampling
                if self.draft_model_temp > 0:
                    step_cand_probs = norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p) # 1 * vocab_size

                    cand_tokens = torch.multinomial(
                        step_cand_probs,
                        self.beam_width,
                        replacement = self.replacement,
                        ).view(-1,1)
                else:
                    topk_logit, topk_index = logits.topk(
                        k=self.beam_width, dim=-1
                    )  # seq_len x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    step_cand_probs = torch.zeros_like(logits)
                    step_cand_probs.scatter_(dim=1, index=topk_index, src=topk_probs)
                    cand_tokens = topk_index.view(-1, 1)

                log_beam_prob = step_cand_probs[:,cand_tokens].view(self.beam_width,1).log_softmax(dim=0)
                cand_probs = step_cand_probs[:,cand_tokens].view(1,-1).log()

                self.draft_p_history.append(step_cand_probs)

                #token_dist_hist = step_cand_probs.repeat_interleave(self.beam_width, dim=0)[:,None,:]
                # modify input_ids based on sampling results
                input_ids = torch.cat(
                  (
                    input_ids,
                    cand_tokens.view(1,-1),
                  ),
                  dim=1,
                )

                # update tree attention
                tree_att_mask[:self.beam_width, input_len:input_len+self.beam_width] = torch.eye(self.beam_width, dtype=torch.bool)


                # modify key value cache 
                #if step + 1 != self.max_draft_len:
                #    for i in range(len(past_key_values)):
                #        past_key_values[i] = (
                #            past_key_values[i][0].repeat_interleave(self.beam_width, dim=0),
                #            past_key_values[i][1].repeat_interleave(self.beam_width, dim=0),
                #        )

                 

            else:
                """ log_beam_prob shape should be (num_beams, 1) """
                """ logits shape should be (num_beams, num_vocab) """
                logits = self.draft_model.lm_head(hidden_states[:, -self.beam_width:]).view(self.beam_width, -1)

                # do sampling
                if self.draft_model_temp > 0:
                    # compute beam probability
                    step_cand_probs = torch.log(norm_logits(logits, self.draft_model_temp, self.top_k, self.top_p))

                    vocab_size = step_cand_probs.size(-1)
                    beam_probs = (log_beam_prob + step_cand_probs).view(-1).exp()

                    cand_beams = torch.multinomial(
                        beam_probs,
                        self.beam_width,
                        replacement = self.replacement,
                        ).view(-1,1)
                else:
                    step_cand_probs = torch.log_softmax(logits, dim=-1)
                    vocab_size = step_cand_probs.size(-1)
                    beam_probs = (log_beam_prob + step_cand_probs).view(-1).exp()

                    topk_logit, topk_index = beam_probs.topk(
                        k=self.beam_width, dim=-1
                    )  # seq_len x k
                    topk_probs = torch.softmax(topk_logit, dim=-1)
                    beam_probs = torch.zeros_like(beam_probs)
                    beam_probs.scatter_(dim=0, index=topk_index, src=topk_probs)
                    cand_beams = topk_index.view(-1, 1)

                self.draft_p_history.append(step_cand_probs.exp())


                log_beam_prob = beam_probs[cand_beams].log_softmax(dim=0) # shape should be (beam_width, 1)

                # modify input_ids based on sampling results
                beam_idx = torch.div(cand_beams, vocab_size, rounding_mode = 'floor').long().view(-1)
                self.beam_idx_history.append(beam_idx)
                tokens = cand_beams % vocab_size


                # cand_probs is used for verification, TODO need to change it
                log_token_probs = step_cand_probs.view(-1)
                log_sampled_token_probs = log_token_probs[cand_beams] # shape (beam_width, 1)
                cur_cand_probs = cand_probs[-1, beam_idx] + log_sampled_token_probs.view(-1)
                #print(cur_cand_probs.exp())
                cand_probs = torch.cat((cand_probs, cur_cand_probs.view(1,-1)), dim=0)

                #token_dist_hist = token_dist_hist[beam_idx]
                #cur_token_dist = step_cand_probs.exp()[beam_idx][:, None,:]
                #token_dist_hist = torch.cat((token_dist_hist, cur_token_dist), dim=1)

                # update tree attention
                tree_att_mask[bias:self.beam_width+bias] = tree_att_mask[bias-self.beam_width: bias][beam_idx.cpu()]
                tree_att_mask[bias:self.beam_width+bias, bias+input_len:bias+input_len+self.beam_width] = torch.eye(self.beam_width, dtype=torch.bool)


                input_ids = torch.cat(
                  (
                    input_ids,
                    tokens.view(1,-1),
                  ),
                  dim=1,
                )




                # modify key value cache
                #for i in range(len(past_key_values)):
                #    if step + 1 < self.max_draft_len:
                #        past_key_values[i] = (
                #            past_key_values[i][0][beam_idx],
                #            past_key_values[i][1][beam_idx],
                #            )
                #    else:
                #        past_key_values[i] = (
                #            past_key_values[i][0][beam_idx:beam_idx+1],
                #            past_key_values[i][1][beam_idx:beam_idx+1],
                #            )

        self.tree_att_mask = tree_att_mask
        return DecoderOnlyDraftOutput(
            sequences=input_ids,
            past_key_values=past_key_values,
            cand_probs=tuple(cand_probs),
        )

    def _forward_target_model(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
    ):
        input_ids = input_ids.to(self.target_model_device)
        #tree_attn_len = self.tree_attn_self_mask.size(0)
        init_input_length = input_ids.size(1) - self.num_token_per_iter
        init_forward = False

        if past_key_values is not None:
            pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = input_ids
            init_forward = True

        if init_forward:
            
            tree_attn_mask = torch.tril(torch.ones(input_ids.size(1), input_ids.size(1), dtype=torch.bool))
            tree_attn_mask[-self.num_token_per_iter:] = self.tree_att_mask
            tree_attn_mask = tree_attn_mask.to(self.target_model_device)
            position_ids = tree_attn_mask.sum(dim=1) - 1
            #print(tree_attn_mask[-8:,-10:])

        else:
            tree_attn_mask = torch.ones(
                (
                    self.num_token_per_iter + 1,
                    input_ids.size(1),
                ),  # there is one token not stored in the kv values
                dtype=torch.bool,
                device=self.target_model_device,
            )

            tree_attn_mask[1:] = self.tree_att_mask
            tree_attn_mask[0, init_input_length:] = 0

            position_ids = tree_attn_mask.sum(dim=1) - 1
        #print(tree_attn_mask[-self.num_token_per_iter:, -self.num_token_per_iter:])
        #print(position_ids)

        outputs: BaseModelOutputWithPast = self.target_model.model(
            input_ids=pruned_input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            tree_attn_mask=tree_attn_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state
        past_key_values = list(outputs.past_key_values)

        logits = self.target_model.lm_head(
            hidden_states[:, -self.num_token_per_iter - 1 :]
        )  # 1 x seq_len x hidden_dim
        return logits, past_key_values

    def verify(
        self,
        input_ids: torch.LongTensor,
        target_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        draft_model_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]],
        cand_probs: Optional[Tuple[torch.FloatTensor]],
    ) -> DecoderOnlyVerificationOutput:

        input_ids = input_ids.to(self.target_model_device)
        input_len = input_ids.size(1)
        logits, target_model_past_key_values = self._forward_target_model(
            input_ids, target_model_past_key_values
        )


        if self.target_model_temp == 0:
            _, topk_index = logits.topk(k=1, dim=-1)  # seq_len x 1
            ground_probs = torch.zeros_like(logits)
            ground_probs.scatter_(dim=2, index=topk_index, value=1)
        else:
            ground_probs = norm_logits(logits, self.target_model_temp, self.top_k, self.top_p)

        unverified_input_ids = input_ids[:, -self.num_token_per_iter :]

        assert ground_probs.size(1) == unverified_input_ids.size(1) + 1

        cand_probs_idx = 0
        alive_group_id = 0

        log_cum_target_prob = 0
        log_cum_draft_prob = 0
        acc_len = 0

#        cand_probs, token_dist_hist = cand_probs
#        cand_probs = tuple(cand_probs)
        best_idx_list = []

        for depth in range(self.max_draft_len):
            #print('depth', depth)

            log_cum_draft_prob = cand_probs[depth]
            bias = depth * self.beam_width
            tokens = unverified_input_ids[0, bias:bias+self.beam_width]
            if depth == 0:
                log_cum_target_prob = ground_probs[0, 0, tokens].log()
                target_p = ground_probs[0,0] 
            else:
                cur_ground_probs = ground_probs[0, 1+bias-self.beam_width:1+bias]
                cur_beam_idx = self.beam_idx_history[depth-1]
                target_token_prob = cur_ground_probs[cur_beam_idx, tokens]
                log_cum_target_prob = log_cum_target_prob[cur_beam_idx] + target_token_prob.log()
                target_p = cur_ground_probs

            draft_p = self.draft_p_history[depth]
            #print(draft_p.size(), target_p.size())
            #print(draft_p.max(), draft_p.sum())
            #print(target_p.max(), target_p.sum())
            #print((draft_p-target_p).abs().sum())


            #print('1')
            #print(log_cum_draft_prob.exp())
            #print(2)
            #print(log_cum_target_prob.exp())
            #xx = input("pause")
                
            best_log_p = None
            best_idx = None
            for i in range(self.beam_width):
                log_ratio = log_cum_target_prob[i] - log_cum_draft_prob[i]
                if log_ratio > self.log_accept_thres:
                    acc_len = depth + 1
                    if (best_log_p is None) or (log_cum_target_prob[i] > best_log_p):
                        best_log_p = log_cum_target_prob[i]
                        best_idx = i
                else:
                    pass
            best_idx_list.append(best_idx)

        if acc_len == 0:
            select_idx = torch.ones((input_len), dtype=torch.bool)
            select_idx[-self.num_token_per_iter] = False
            accept_beam_pos = -1
        else:
            accept_beam_pos = (acc_len - 1) * self.beam_width + best_idx_list[acc_len-1]
            select_idx = self.tree_att_mask[accept_beam_pos]

        draft_select_idx = select_idx[:-self.beam_width]

        input_ids = input_ids[:, select_idx]
        endpoint_token = torch.multinomial(
                ground_probs[:, accept_beam_pos+1], num_samples=1
                ).to(device=input_ids.device)

        input_ids = torch.cat((input_ids, endpoint_token), dim=-1)

        for i in range(len(target_model_past_key_values)):
            target_model_past_key_values[i] = (
                    target_model_past_key_values[i][0][:, :, select_idx],
                    target_model_past_key_values[i][1][:, :, select_idx],
                    )

        for i in range(len(draft_model_past_key_values)):
            draft_model_past_key_values[i] = (
                    draft_model_past_key_values[i][0][:, :, draft_select_idx],
                    draft_model_past_key_values[i][1][:, :, draft_select_idx],
                    )


        return DecoderOnlyVerificationOutput(
            sequences=input_ids,
            target_model_past_key_values=target_model_past_key_values,
            draft_model_past_key_values=draft_model_past_key_values,
            acceptance_count=acc_len,
        )

