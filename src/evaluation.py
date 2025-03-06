import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
hf_token = os.environ['HFTOKEN']

import argparse
import json
import logging
import time
from typing import Literal, Tuple

import torch
from inference.generate import Generator, BaseGenerator, SpeculativeGenerator, MTADGenerator, DSBDGenerator
from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
import numpy as np
from utils import find_fields_MYSQL_like, creating_schema, spider_examples
from utils import execution_accuracy_references, extract_first_function
from utils import get_score, get_total_power
import subprocess

def set_seed(seed: int):
    random.seed(seed)                      # Python random module
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                 # PyTorch CPU
    torch.cuda.manual_seed(seed)            # PyTorch GPU (single-GPU)
    torch.cuda.manual_seed_all(seed)        # PyTorch GPU (multi-GPU)
                       
    # Ensure deterministic behavior in PyTorch operations (if possible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class JsonData:
    def __init__(self, path) -> None:
        with open(path) as fin:
            self.data = json.load(fin)
            self.data = self.data[:100]

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def run_eval(
    draft_model,
    target_model,
    tokenizer,
    dataset,
    dataloader,
    k_config: Tuple[int],
    max_new_tokens: int = 128,
    replacement=False,
    speculative_sampling=True,
    tree_attn=True,
    sampling_type: Literal["argmax", "sampling"] = "sampling",
    disable_tqdm: bool = False,
    mtad: bool = True,
    dsbd: bool = False,
    beam_width: int = 4,
    accept_thres: float = 0.5,
    expect_thres: float = 0.8,
    min_accept_num: int = 1,
    top_k: int = 10,
    top_p: float = 0.9,
):
    if sampling_type not in ["argmax", "sampling"]:
        raise ValueError(
            f'`sampling_type` can be either `"argmax"` or `"sampling"`, but received "{sampling_type}"'
        )
    if sampling_type == "argmax":
        target_model_temp = 0
        draft_model_temp = 0
    else:
        target_model_temp = 1
        draft_model_temp = 1

    if mtad:
      generator = MTADGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config,
        beam_width = beam_width,
        accept_thres = accept_thres,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=replacement,
        speculative_sampling=speculative_sampling,
        tree_attn=tree_attn,
        top_k = top_k,
        top_p = top_p,
      )
    elif dsbd:
       generator = DSBDGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config,
        beam_width = beam_width,
        min_accept_num = min_accept_num,
        expect_thres = expect_thres,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=replacement,
        speculative_sampling=speculative_sampling,
        top_k = top_k,
        top_p = top_p,
      )
       
    else:
      generator = SpeculativeGenerator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=replacement,
        speculative_sampling=speculative_sampling,
        tree_attn=tree_attn,
        top_k = top_k,
        top_p = top_p,
      )

    draft_model.eval()
    target_model.eval()

    logger.info("evaluation start.")
    start_time = time.time()

    acceptance_count = 0
    draft_token_count = 0
    invocation_count = 0

    iterator = range(len(dataloader))
    pred_seq = []
    score_list = []

    P = subprocess.Popen("exec python3 -u gpu_power_monitor.py",shell=True, text=True, stdout=subprocess.PIPE)

    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            prompt_text = dataloader[sample_idx]
            inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
            input_ids = inputs.input_ids
            input_len = input_ids.size(-1)
            output = generator.generate(input_ids)

            acceptance_count += output.acceptance_count
            draft_token_count += output.draft_token_count
            invocation_count += output.invocation_count

            if dataset == 'spider':
                pred_seq.append(tokenizer.decode(output.sequences[0][input_len:], skip_special_tokens=True).split(';')[0])
            elif dataset == 'human_eval':
                string = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
                pred_seq.append(extract_first_function(string))

            score = get_score(output.sequences, target_model, input_len)
            score_list.append(score.item())

    end_time = time.time()
    P.kill()
    P.wait()

    logger.info("evaluation complete.")

    run_time = end_time - start_time

    latency = run_time / (acceptance_count + invocation_count)
    acceptance_rate = acceptance_count / draft_token_count
    block_efficiency = 1 + acceptance_count / invocation_count

    outputs = P.stdout.readlines()
    power_total = get_total_power(outputs, start_time, end_time, None)


    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))
    logger.info("Acceptance rate: {:.2f}".format(acceptance_rate))
    logger.info("Block efficiency: {:.2f}".format(block_efficiency))
    logger.info("J/token: {:.2f}".format(power_total/(acceptance_count+invocation_count)))
    logger.info("PPL: {:.2f}".format(np.exp(-np.mean(score_list))))
    return pred_seq


def run_baseline_eval(
    target_model,
    tokenizer,
    dataloader,
    max_new_tokens: int = 128,
    sampling_type: Literal["argmax", "sampling"] = "sampling",
    disable_tqdm: bool = False,
    top_k: int = 10,
    top_p: float = 0.9,
):
    if sampling_type not in ["argmax", "sampling"]:
        raise ValueError(
            f'`sampling_type` can be either `"argmax"` or `"sampling"`, but received "{sampling_type}"'
        )
    if sampling_type == "argmax":
        target_model_temp = 0
    else:
        target_model_temp = 1

    generator = BaseGenerator(
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temp=target_model_temp,
        top_k = top_k,
        top_p = top_p,
    )

    target_model.eval()

    logger.info("evaluation start.")
    start_time = time.time()

    invocation_count = 0

    iterator = range(len(dataloader))
    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            prompt_text = dataloader[sample_idx]
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            output = generator.generate(input_ids)

            invocation_count += output.invocation_count
    end_time = time.time()

    logger.info("evaluation complete.")

    run_time = end_time - start_time

    latency = run_time / invocation_count

    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))


def main(args):
    if args.mtad == True and args.dsbd == True:
        logger.warning(
            "When both --mtad and --dsbd flags are set, only running mtad."
        )
        args.dsbd = False


    set_seed(args.seed)  # Set a fixed seed
    torch_dtype = torch.float16 if args.fp16 else torch.float32

    logger.info("The full evaluation configuration:\n" + repr(args))

    if args.auto_model and not args.disable_tree_attn:
        logger.warning(
            "Tree Attn is currently not supported for models other than LLaMA. Therefore, "
            "when using '--auto-model', Tree Attn will be disabled."
        )
        args.disable_tree_attn = True

    if args.dataset == 'human_eval':
        from human_eval.data import write_jsonl, read_problems
        problems = read_problems()
        prefix = ""
        postfix = ""
        dataloader = [
                      problems[task_id]["prompt"]  
                      for task_id in problems
        ]

    elif args.dataset == 'spider':
        dataset = json.load(open(os.path.join(args.datapath, "dev.json")))
        spider_schema,spider_primary,spider_foreign = creating_schema(os.path.join(args.datapath, "tables.json"))

        dataloader = [spider_examples + 
                       "Schema:\n" + find_fields_MYSQL_like(s["db_id"], spider_schema) + "\n" + 
                       "Question: " + s["question"] + "\n" + 
                       "SQL:" for s in dataset]
        output_dataset = [s["db_id"] + "[SQL]" + s["query"] for s in dataset] 
        dataloader = dataloader[:100]

    else:
        raise NotImplementedError

    ModelLoader = AutoModelForCausalLM if args.auto_model else LlamaForCausalLM
    TokenizerLoader = AutoTokenizer if args.auto_model else LlamaTokenizer

    print(args.tokenizer, type(args.tokenizer))
    print(hf_token, type(hf_token))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code = True, token = hf_token)

    logger.info("Loading draft model: {}".format(args.draft_model))
    draft_model = ModelLoader.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map=0,
        use_flash_attention_2=True if args.flash_attn else False,
        token = hf_token,
        cache_dir = '/llmss/cache/huggingface/',
    )

    logger.info("Loading target model: {}".format(args.target_model))
    target_model = ModelLoader.from_pretrained(
        args.target_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        use_flash_attention_2=True if args.flash_attn else False,
        token = hf_token,
        max_memory={0: "38GiB"},
        cache_dir = '/llmss/cache/huggingface/',

    )


    if args.run_baseline:
        run_baseline_eval(
            target_model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            max_new_tokens=args.max_new_tokens,
            sampling_type=args.sampling_type,
            disable_tqdm=args.disable_tqdm,
            top_k = args.top_k,
            top_p = args.top_p,
        )
    else:
        pred_seq = run_eval(
            draft_model,
            target_model,
            tokenizer=tokenizer,
            dataset=args.dataset,
            dataloader=dataloader,
            k_config=args.k_config,
            beam_width=args.beam_width,
            accept_thres=args.accept_thres,
            max_new_tokens=args.max_new_tokens,
            replacement=args.replacement,
            speculative_sampling=not args.naive_sampling,
            tree_attn=not args.disable_tree_attn,
            sampling_type=args.sampling_type,
            disable_tqdm=args.disable_tqdm,
            mtad = args.mtad,
            dsbd = args.dsbd,
            expect_thres = args.expect_thres,
            min_accept_num = args.min_accept_num,
            top_k = args.top_k,
            top_p = args.top_p,
        )
        if args.dataset == "spider":
            cnt = len(pred_seq)
            performance = execution_accuracy_references(predictions = pred_seq, references = output_dataset[:cnt], data_path = args.datapath)
            print(performance)
        elif args.dataset == 'human_eval':
            samples = [dict(task_id=task_id, completion=output) for task_id, output in zip(problems,pred_seq)]
            file_name = f"output_{time.perf_counter()}.jsonl"
            write_jsonl(file_name, samples)
            print(f"write output to {file_name}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="dataset: spider or human_eval")
    parser.add_argument(
        "--draft-model", type=str, required=True, help="Draft model path."
    )
    parser.add_argument(
        "--target-model", type=str, required=True, help="Target model path."
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--fp16", action="store_true", help="use float16 dtype.")

    parser.add_argument(
        "--k-config",
        type=lambda x: tuple(map(int, x.split(","))),
        required=True,
        help="Use comma separations, e.g. `--k-config 4,2,2`.",
    )

    parser.add_argument(
        "--datapath", type=str, required=True, help="The json data file."
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--replacement",
        action="store_true",
        help="Sampling with replacement.",
    )
    parser.add_argument(
        "--naive-sampling",
        action="store_true",
        help="Use multi-candidate naive sampling.",
    )

    parser.add_argument("--disable-tree-attn", action="store_true")

    parser.add_argument(
        "--sampling-type", type=str, default="sampling", choices=["argmax", "sampling"]
    )

    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--auto-model", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")

    parser.add_argument("--flash-attn", action="store_true")
    # mtad parameters
    parser.add_argument("--mtad", action="store_true")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--accept-thres", type=float, default=0.5)

    # dsbd parameters
    parser.add_argument("--dsbd", action="store_true")
    parser.add_argument("--min-accept-num", type=int, default=1)
    parser.add_argument("--expect-thres", type=float, default=0.8)

    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    main(args)
