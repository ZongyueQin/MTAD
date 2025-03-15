from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from . import strategies
from utils import norm_logits
import time


@dataclass
class DecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using MCSD.
    """

    sequences: torch.LongTensor
    acceptance_count: int = None
    draft_token_count: int = None
    invocation_count: int = None


class Generator:
    def __init__(self) -> None:
        pass

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        raise NotImplementedError


class BaseGenerator:
    def __init__(
        self,
        model,
        eos_token_id: int,
        max_new_tokens: int = 128,
        temp: float = 1,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        past_key_values = None
        invocation_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids

            outputs: CausalLMOutputWithPast = self.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            batch_num, seq_len, _ = logits.size()

            if self.temp == 0:
                _, ground_tokens = logits.topk(k=1, dim=-1)  # batch x seq_len x 1
            else:
                ground_probs = norm_logits(logits, self.temp, self.top_k, self.top_p) 

                ground_tokens = torch.multinomial(
                    ground_probs.view(batch_num * seq_len, -1), num_samples=1
                )  # batch*seq_len x 1
            ground_tokens = ground_tokens.view(batch_num, seq_len)

            input_ids = torch.cat(
                (input_ids, ground_tokens[:, -1:].to(input_ids)), dim=1
            )

            invocation_count += 1

            if (
                self.eos_token_id in input_ids[0, -1:]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(sequences=input_ids, invocation_count=invocation_count)


class SpeculativeGenerator:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        tree_attn: bool = True,
        mtad: bool = True,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None
        self.draft_time = 0
        self.verify_time = 0
        self.other_time = 0

        if tree_attn:
            self.strategy = strategies.TreeStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
                top_k = top_k,
                top_p = top_p,
            )
        else:
            self.strategy = strategies.BatchStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
                top_k = top_k,
                top_p = top_p,
            )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        target_model_past_key_values = None
        draft_model_past_key_values = None

        invocation_count = 0
        acceptance_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            start_time = time.time()
            draft_output = self.strategy.generate_draft(
                input_ids,
                past_key_values=draft_model_past_key_values,
            )

            draft_model_past_key_values = draft_output.past_key_values
            self.draft_time += time.time() - start_time

            start_time = time.time()

            verification_output = self.strategy.verify(
                input_ids=draft_output.sequences,
                target_model_past_key_values=target_model_past_key_values,
                draft_model_past_key_values=draft_output.past_key_values,
                cand_probs=draft_output.cand_probs,
            )
            self.verify_time += time.time() - start_time

            start_time = time.time()

            input_ids = verification_output.sequences

            draft_model_past_key_values = (
                verification_output.draft_model_past_key_values
            )
            target_model_past_key_values = (
                verification_output.target_model_past_key_values
            )

            invocation_count += 1
            acceptance_count += verification_output.acceptance_count
            self.other_time += time.time() - start_time

            if (
                self.eos_token_id in input_ids[0, -self.strategy.max_draft_len :]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(
            sequences=input_ids,
            acceptance_count=acceptance_count,
            draft_token_count=invocation_count * self.strategy.max_draft_len,
            invocation_count=invocation_count,
        )

class MTADGenerator(SpeculativeGenerator):
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        beam_width: int = 4,
        accept_thres: float = 0.5,
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        tree_attn: bool = True,
        mtad: bool = True,
        v2: bool = False,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None
        self.draft_time = 0
        self.verify_time = 0
        self.other_time = 0


        if tree_attn == False:
            self.strategy = strategies.BatchMTADStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                beam_width=beam_width,
                accept_thres=accept_thres,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
                top_k = top_k,
                top_p = top_p,
            )
        else:
            if v2 == False:
                self.strategy = strategies.TreeMTADStrategy(
                  draft_model=draft_model,
                  target_model=target_model,
                  k_config=k_config,
                  beam_width=beam_width,
                  accept_thres=accept_thres,
                  draft_model_temp=draft_model_temp,
                  target_model_temp=target_model_temp,
                  replacement=replacement,
                  speculative_sampling=speculative_sampling,
                  top_k = top_k,
                  top_p = top_p,
                )
            else:
                self.strategy = strategies.TreeMTADStrategy_v2(
                  draft_model=draft_model,
                  target_model=target_model,
                  k_config=k_config,
                  beam_width=beam_width,
                  accept_thres=accept_thres,
                  draft_model_temp=draft_model_temp,
                  target_model_temp=target_model_temp,
                  replacement=replacement,
                  speculative_sampling=speculative_sampling,
                  top_k = top_k,
                  top_p = top_p,
                )


class DSBDGenerator(SpeculativeGenerator):
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        beam_width: int = 4,
        min_accept_num: int = 1,
        expect_thres: float = 0.8,
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None
        self.draft_time = 0
        self.verify_time = 0
        self.other_time = 0


        self.strategy = strategies.SingleDSBDStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                beam_width=beam_width,
                min_accept_num = min_accept_num,
                expect_thres=expect_thres,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
                top_k = top_k,
                top_p = top_p,
            )

