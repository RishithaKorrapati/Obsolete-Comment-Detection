# Obsolete Comment Detection (Capstone)

Detect **comment–code inconsistency (CCI)** in Java: given a **code snippet** and a **documentation comment** (summary, `@param`, or `@return` style), predict whether they are **consistent** or **inconsistent**.

This repository extends the supplementary pipeline from [*Code Comment Inconsistency Detection and Rectification Using a Large Language Model*](https://github.com/aiopsplus/C4RLLaMA) with a **memory-efficient local model** (**TinyLlama 1.1B + LoRA**) and **cloud API baselines** (**GPT-4o-mini**, **Gemini**).

**Advisor:** Dr. Zhe Yu

---

## What we built

| Track | Approach | Role |
|--------|-----------|------|
| **Local** | Fine-tune **TinyLlama/TinyLlama-1.1B-Chat-v1.0** with **LoRA** on ~4 GB GPU | Trainable detector under hardware constraints |
| **Cloud** | **GPT-4o-mini** / **Gemini** with zero-shot and few-shot prompts | API baselines without local weight updates |

**Task:** Binary classification — label **0** = consistent, **1** = inconsistent.

**Training:** `BalanceTrainer` with **classification-aware label smoothing** (`classification_alpha`, `label_smoothing_factor`).

**Inference (local):** Instruction prompt → short generation → heuristic parse (`"inconsisten"` in output → inconsistent).

---

## Results (logged runs)

### Local — TinyLlama + LoRA (`result.txt`)

| Run | Accuracy | F1 |
|-----|----------|-----|
| 1 | 0.7253 | 0.7211 |
| 2 (best) | **0.7732** | **0.7661** |

### API — GPT-4o-mini (`result_openai.txt`, accuracy)

| Prompt | Few-shot k | N | Accuracy |
|--------|------------|---|----------|
| new code + comment | 0 | 30 | 0.7667 |
| new code + comment | 4 | 50 | 0.7200 |
| new code + comment | 0 | 150 | 0.6533 |
| new code + comment | 0 | 600 | 0.6667 |
| old + new code + comment | 0 | 50 | 0.6400 |
| old + new code + comment | 4 | 50 | 0.6600 |

**Takeaway:** A **small locally fine-tuned** model reaches **~77% accuracy** on full local evaluation, competitive with strong **GPT-4o-mini** runs on smaller subsets, while API scores vary with **N** and **prompt style**.

---

## Repository layout

```
.
├── train.py              # LoRA fine-tuning (TinyLlama / other HF causal LMs)
├── test.py               # Local evaluation on Data/{Summary,Param,Return}/test.json
├── eval_openai.py        # GPT-4o-mini zero-shot / few-shot evaluation
├── eval_gemini.py        # Gemini zero-shot / few-shot evaluation
├── utils/
│   ├── BalanceTrainer.py # Trainer + classification label smoothing
│   └── prompter.py       # Instruction templates
├── templates/
│   └── llama.json        # Prompt template config
├── Data.7z               # Dataset archive (extract to Data/)
├── result.txt            # Local run metrics log
├── result_openai.txt     # OpenAI run metrics log
├── result_gemini.txt     # Gemini run metrics log
├── requirements.txt
└── LICENSE.txt
```

**Not in git (see `.gitignore`):** `.env`, `.venv/`, extracted `Data/`, `LoraTinyLlama_1.1B/` (train locally or host weights separately).

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/RishithaKorrapati/Obsolete-Comment-Detection.git
cd Obsolete-Comment-Detection
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Extract the dataset:

```bash
# Unzip Data.7z into ./Data/
```

Expected structure:

```
Data/
├── Summary/train.json  test.json  valid.json
├── Param/train.json    test.json  valid.json
└── Return/train.json   test.json  valid.json
```

Each JSON record includes fields such as `new_code_raw`, `old_comment_raw`, `label` (and optionally `old_code_raw`, `new_comment_raw`).

### 3. API keys (optional, for cloud baselines)

Create `.env` in the project root:

```env
CHATGPT_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

Never commit `.env`.

### 4. LoRA weights (local inference)

After training, adapters are saved under `./LoraTinyLlama_1.1B/` (gitignored). Place your checkpoint there or pass `--lora_weights` to `test.py`.

---

## Usage

### Train (TinyLlama + LoRA)

Adjust paths and batch sizes for your GPU. Example:

```bash
python -u train.py ^
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --data_path Data/LLMtrainDataset.jsonl ^
  --output_dir ./LoraTinyLlama_1.1B ^
  --micro_batch_size 1 ^
  --batch_size 4 ^
  --num_epochs 3 ^
  --learning_rate 1e-4 ^
  --cutoff_len 512 ^
  --val_set_size 100 ^
  --prompt_template_name llama ^
  --label_smoothing_factor 0.1 ^
  --classification_alpha 0.5 ^
  --train_on_inputs False
```

### Test locally (fine-tuned model)

```bash
python -u test.py ^
  --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ^
  --lora_weights ./LoraTinyLlama_1.1B ^
  --prompt_template llama
```

Metrics append to `result.txt`; per-example outputs go to `Data/TestResult.xlsx` by default.

### Evaluate GPT-4o-mini

Zero-shot on all categories (cap total examples for cost):

```bash
python -u eval_openai.py --model_name gpt-4o-mini --few_shot_k 0 --max_total_examples 30
```

Few-shot (k=4), change-aware prompt:

```bash
python -u eval_openai.py --model_name gpt-4o-mini --few_shot_k 4 --max_total_examples 50 --prompt_style old_new_code_comment
```

Logs append to `result_openai.txt`.

### Evaluate Gemini

```bash
python -u eval_gemini.py --model_name gemini-2.0-flash --few_shot_k 0
```

Logs append to `result_gemini.txt`.

---

## Prompt styles (API)

| `prompt_style` | Inputs |
|----------------|--------|
| `new_code_comment` (default) | Current code + comment under audit |
| `old_new_code_comment` | Previous code + current code + comment |

Few-shot demonstrations are sampled from **`train.json` only**, never from test data.

---

## Acknowledgments

- Based on the **C4RLLaMA** supplementary codebase and Java CCI dataset from the original paper authors.
- Capstone work advised by **Dr. Zhe Yu**.

## License

See [LICENSE.txt](LICENSE.txt) (Apache 2.0).
