# Multi-Token Assisted Decoding (MTAD)

This repository contains the implementation of **MTAD** from the paper:
**"Optimized Multi-Token Joint Decoding with Auxiliary Model for LLM Inference"**, ICLR 2025.

The implementation is based on [MCSD](https://github.com/NJUNLP/MCSD).

---

## Update

**2025.4.9**: Implement Multi-Candidate MTAD, which incorporates tree-wise parallel decoding for better efficiency and output quality. The details of the algorithm will be released on arxiv. 

## 🚀 Dependencies
Ensure you have the following installed:

- **PyTorch**: `>= 2.4.1`
- **Python**: `>= 3.8`
- **Transformers**: `>= 4.34.0`
- **pandas**

---

## 📂 Datasets

### Spider
Download the **Spider** dataset from their official website:
[https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)

### Human-Eval
Install **Human-Eval** from its GitHub repository:
[https://github.com/openai/human-eval](https://github.com/openai/human-eval)

### MT-Bench
The script does not directly support **MT-Bench**, but you can modify the script from [FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py) to generate answers using our decoding method and run evaluation.

---

## 🛠 Usage

### Setting up the Environment
If you want to run official **Llama Models**, set your Hugging Face token first:
```sh
env HFTOKEN=your_huggingface_token
```
Then, run `evaluation.py` with the appropriate options.

### Important Options

| Argument | Description |
|----------|-------------|
| `--dataset` | Name of the dataset (`spider` or `human_eval`) |
| `--draft-model` | Path to the draft model |
| `--target-model` | Path to the target model |
| `--tokenizer` | Path to the tokenizer (defaults to target model if not provided) |
| `--mtad` | Run MTAD decoding |
| `--beam-width` | Beam width of the draft model for MTAD (default: `4`) |
| `--accept-thres` | Acceptance threshold for MTAD (default: `0.5`) |
| `--fp16` | Use float16 dtype for the target model |
| `--k-config` | Branch factor for SpecInfer (comma-separated values, e.g., `--k-config 4,2,2`) |
| `--datapath` | Path to the JSON data file |
| `--max-new-tokens` | Maximum number of new tokens |
| `--replacement` | Enable sampling with replacement |
| `--disable-tqdm` | Disable tqdm progress bar |

---

## 📌 Example Commands and Outputs

For detailed example scripts and outputs, refer to [examples.md](examples.md).

---

## ⚠️ Notes
- **SpecInfer** utilizes **tree attention**, which is only implemented for the **Llama model**.
- **MTAD does not require tree attention**, so you can directly use `AutoModelForCausalLM` with MTAD.

---

## 🔗 References
- [MCSD Repository](https://github.com/NJUNLP/MCSD)
- [Spider Dataset](https://yale-lily.github.io/spider)
- [Human-Eval Repository](https://github.com/openai/human-eval)
- [FastChat MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py)

---

Now, you're all set to use **MTAD** for efficient LLM inference! 🚀

