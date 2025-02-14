## Example Commands and Outputs

### Spider Dataset

#### Llama-3 MTAD
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.1-8B --max-new-tokens 100 --k-config 1,1,1,1 --datapath PATH_TO_SPIDER --dataset spider --replacement --mtad --accept-thres 0.5
```

**Output**:
```
02/13/2025 10:20:49 - INFO - __main__ - evaluation complete.
02/13/2025 10:20:49 - INFO - __main__ - Running time: 326.39 s
02/13/2025 10:20:49 - INFO - __main__ - Token latency: 33.20 ms
02/13/2025 10:20:49 - INFO - __main__ - Acceptance rate: 0.80
02/13/2025 10:20:49 - INFO - __main__ - Block efficiency: 4.19
{'execution accuracy': 47.0, 'exception': 0}
```

#### Llama-3 SpecInfer
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.1-8B --max-new-tokens 100 --k-config 4,2,2 --datapath PATH_TO_SPIDER --dataset spider --replacement
```

**Output**:
```
02/13/2025 10:45:25 - INFO - __main__ - evaluation complete.
02/13/2025 10:45:25 - INFO - __main__ - Running time: 327.93 s
02/13/2025 10:45:25 - INFO - __main__ - Token latency: 35.52 ms
02/13/2025 10:45:25 - INFO - __main__ - Acceptance rate: 0.85
02/13/2025 10:45:25 - INFO - __main__ - Block efficiency: 3.55
{'execution accuracy': 37.0, 'exception': 0}
```

#### Llama-3-Instruct MTAD
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B-Instruct --target-model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 100 --k-config 1,1,1,1 --datapath PATH_TO_SPIDER --dataset spider --replacement --mtad --accept-thres 0.3
```

**Output**:
```
02/13/2025 12:53:50 - INFO - __main__ - evaluation complete.
02/13/2025 12:53:50 - INFO - __main__ - Running time: 305.81 s
02/13/2025 12:53:50 - INFO - __main__ - Token latency: 30.04 ms
02/13/2025 12:53:50 - INFO - __main__ - Acceptance rate: 0.78
02/13/2025 12:53:50 - INFO - __main__ - Block efficiency: 4.14
{'execution accuracy': 64.0, 'exception': 0}
```

#### Llama-3-Instruct SpecInfer
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B-Instruct --target-model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 100 --k-config 4,2,2 --datapath PATH_TO_SPIDER --dataset spider --replacement
```

**Output**:
```
02/13/2025 12:47:05 - INFO - __main__ - evaluation complete.
02/13/2025 12:47:05 - INFO - __main__ - Running time: 333.12 s
02/13/2025 12:47:05 - INFO - __main__ - Token latency: 32.91 ms
02/13/2025 12:47:05 - INFO - __main__ - Acceptance rate: 0.79
02/13/2025 12:47:05 - INFO - __main__ - Block efficiency: 3.36
{'execution accuracy': 52.0, 'exception': 0}
```

---

### HumanEval Dataset

To get the Pass@1 of HumanEval, run `evaluate_functional_correctness xxxx.jsonl` command after generating the output file.

#### Llama-3 MTAD
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.1-8B --max-new-tokens 100 --k-config 1,1,1,1 --datapath "" --dataset human_eval --replacement --mtad --accept-thres 0.5
```

**Output**:
```
02/13/2025 14:07:34 - INFO - __main__ - evaluation complete.
02/13/2025 14:07:34 - INFO - __main__ - Running time: 263.31 s
02/13/2025 14:07:34 - INFO - __main__ - Token latency: 29.02 ms
02/13/2025 14:07:34 - INFO - __main__ - Acceptance rate: 0.83
02/13/2025 14:07:34 - INFO - __main__ - Block efficiency: 4.33
{'pass@1': 0.29878048780487804}
```

#### Llama-3 SpecInfer
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B --target-model meta-llama/Llama-3.1-8B --max-new-tokens 100 --k-config 4,2,2 --datapath "" --dataset human_eval --replacement
```

**Output**:
```
02/13/2025 14:06:32 - INFO - __main__ - evaluation complete.
02/13/2025 14:06:32 - INFO - __main__ - Running time: 287.85 s
02/13/2025 14:06:32 - INFO - __main__ - Token latency: 30.73 ms
02/13/2025 14:06:32 - INFO - __main__ - Acceptance rate: 0.89
02/13/2025 14:06:32 - INFO - __main__ - Block efficiency: 3.66
{'pass@1': 0.23780487804878048}
```

#### Llama-3-Instruct MTAD
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B-Instruct --target-model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 100 --k-config 1,1,1,1 --datapath "" --dataset human_eval --replacement --mtad --accept-thres 0.5
```

**Output**:
```
02/13/2025 13:48:23 - INFO - __main__ - evaluation complete.
02/13/2025 13:48:23 - INFO - __main__ - Running time: 461.93 s
02/13/2025 13:48:23 - INFO - __main__ - Token latency: 27.67 ms
02/13/2025 13:48:23 - INFO - __main__ - Acceptance rate: 0.84
02/13/2025 13:48:23 - INFO - __main__ - Block efficiency: 4.34
{'pass@1': 0.4573170731707317}
```

#### Llama-3-Instruct SpecInfer
**Command**:
```bash
python evaluation.py --draft-model meta-llama/Llama-3.2-1B-Instruct --target-model meta-llama/Llama-3.1-8B-Instruct --max-new-tokens 100 --k-config 4,2,2 --datapath "" --dataset human_eval --replacement
```

**Output**:
```
02/13/2025 13:38:51 - INFO - __main__ - evaluation complete.
02/13/2025 13:38:51 - INFO - __main__ - Running time: 489.30 s
02/13/2025 13:38:51 - INFO - __main__ - Token latency: 29.48 ms
02/13/2025 13:38:51 - INFO - __main__ - Acceptance rate: 0.88
02/13/2025 13:38:51 - INFO - __main__ - Block efficiency: 3.64
{'pass@1': 0.4268292682926829}
```
