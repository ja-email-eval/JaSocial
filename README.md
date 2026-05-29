# JaSocial
A 3-stage cascaded BERT classifier for scoring honorific language (*keigo*) usage in Japanese emails.

- **Stage 1** — Email-level: social relationship & role-pair recognition
- **Stage 2** — Sentence-level: interaction object, role, and sender action recognition
- **Stage 3** — Sentence-level: keigo style classification

---

## Download Trained Models

The trained model weights are hosted on Google Drive:

**[Download models from Google Drive](https://drive.google.com/drive/folders/1NchwYY6CjULUfLyf1aO489HZKXo2a0PR?usp=sharing)**

After downloading, place the files so the directory looks like:

```
  models/
    stage1_<timestamp>/
      stage1_model.bin
    stage2_<timestamp>/
      stage2_model.bin
    stage3_<timestamp>/
      stage3_model.bin
  runs/
```

---

## Installation

```bash
pip install -r requirements.txt
```

> The tokenizer `cl-tohoku/bert-base-japanese-v2` requires `fugashi` and `ipadic`.
> It is downloaded automatically on first run via Hugging Face.

---

## Usage: Score an Email

```bash
python pipeline.py \
  --model_dir ./models \
  --email "社会場面: あなたが提出した奨学金申請書は書類審査を通過しました。1月20日に面接審査が実施される予定です。
件名: 件名なし
宛先: A教授
本文: お世話になっております。先日私が提出した奨学金申請書が書類審査を通過しました。今月20日には面接審査が行われるので、それに出席します。ご了承ください。"
```

> **Tip:** Including the `社会場面: ...` prefix improves accuracy because the models were trained with that context.

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | `./models` | Root directory containing trained model weights |
| `--label_maps` | `./label_maps.json` | Path to label mapping JSON |
| `--output_dir` | `./runs` | Directory to save scoring results |

**Output** — A JSON file saved in `./runs/` containing:
- `final_score`: overall keigo score (0–1)
- `email_level`: predicted social relationship and role pair
- `sentence_level`: per-sentence predictions (objects, roles, actions, styles)

**Example output for the command above** (from a test set email, 学生→教授):

```json
{
  "final_score": 0.9204,
  "email_level": {
    "social_standing": "目下→目上",
    "role_relation": "学生→教授",
    "confidence": 0.9881
  },
  "sentence_level": [
    {"text": "お世話になっております。", "styles": [{"label": "謙譲語+丁寧語", "score": 0.985}]},
    {"text": "先日私が提出した奨学金申請書が書類審査を通過しました。", "styles": [{"label": "丁寧語", "score": 0.994}]},
    ...
  ]
}
```

---

## Usage: Retrain the Classifiers

If you have access to the annotated email dataset, you can retrain the models.

```bash
# Step 1: Build label maps from your data
python classifier.py --mode build_maps --data_dir ./data

# Step 2: Train each stage in order
python classifier.py --mode train_stage1 --data_dir ./data
python classifier.py --mode train_stage2 --data_dir ./data
python classifier.py --mode train_stage3 --data_dir ./data

# Step 3: Run inference on a labeled JSON file
python classifier.py --mode inference --input ./email.json --model_dir ./models/stage3_<timestamp>
```

---

## Dataset

This repository includes **10 sample emails** (`sample_emails.json`) drawn from the held-out test set (random_state=42), covering all 6 role-pair scenarios.
The complete dataset (1,200 emails) is **not publicly distributed** to prevent automated scraping.
To request the full dataset, please contact the authors by email (see below).

### Dataset Overview

| Split | Social Standing | Role Pair | Emails |
|---|---|---|---|
| 同輩 (peer) | sender = receiver | 学生→友人 | 200 |
| 同輩 (peer) | sender = receiver | 従業員→同僚 | 200 |
| 目下→目上 (subordinate→superior) | sender < receiver | 学生→教授 | 200 |
| 目下→目上 (subordinate→superior) | sender < receiver | 従業員→上司 | 200 |
| 目上→目下 (superior→subordinate) | sender > receiver | 教員→学生 | 200 |
| 目上→目下 (superior→subordinate) | sender > receiver | 従業員→部下 | 200 |
| **Total** | | | **1,200** |

Each email is annotated at two levels:
- **Email-level**: social standing (3 classes) and role pair (6 classes)
- **Sentence-level**: interaction object, interaction role, sender action, and keigo style

**Train / Validation / Test split**: 80% / 10% / 10% (random split with seed 42, mixed across all role pairs)

The split is performed at runtime by the training script (`classifier.py`) using `sklearn.train_test_split(random_state=42)` on the full pooled dataset. There is no pre-saved test file; the same seed guarantees a reproducible split.

| Stage | Granularity | Total | Train | Val | Test |
|---|---|---|---|---|---|
| Stage 1 | email-level | 1,200 | 960 | 120 | 120 |
| Stage 2 | sentence-level | 5,805 | 4,644 | 580 | 581 |
| Stage 3 | sentence-level | 5,805 | 4,644 | 580 | 581 |

### Test Set Performance (trained models, 2025-04-07)

**Stage 1** (120 test emails):
- Social standing accuracy: **100.00%**
- Role pair accuracy: **100.00%**

**Stage 2** (581 test sentences):
- Interaction object accuracy: **94.34%**
- Interaction role accuracy: **94.22%**
- Sender action accuracy: **96.06%**
- Average: **94.88%**

**Stage 3** (581 test sentences):
- Keigo style accuracy: **97.86%**
  - インフォーマル: 98.11% / 丁寧語: 93.63% / 謙譲語+丁寧語: 95.52%
  - 尊敬語+丁寧語: 99.83% / 尊敬語+謙譲語+丁寧語: 99.83% / 謙譲語+インフォーマル: 100.00%

### Sample Emails (`sample_emails.json`)

10 emails are provided as a usage example and for quick testing of the pipeline.
Each entry contains the full annotation structure (email-level labels + sentence-level tags).

| # | Social Standing | Role Pair |
|---|---|---|
| 1–2 | 同輩 (peer) | 学生→友人 |
| 3–4 | 同輩 (peer) | 従業員→同僚 |
| 5–6 | 目下→目上 (subordinate→superior) | 学生→教授 |
| 7–8 | 目下→目上 (subordinate→superior) | 従業員→上司 |
| 9 | 目上→目下 (superior→subordinate) | 教員→学生 |
| 10 | 目上→目下 (superior→subordinate) | 従業員→部下 |

To score one of the sample emails with the pipeline:

```bash
python3 -c "
import json
with open('sample_emails.json') as f:
    emails = json.load(f)
sample = emails[4]  # 学生→教授, index 4
scene = sample['社会場面']
subject = sample['件名'].strip()
receiver = sample['受信者呼び名'].strip()
body = ' '.join(s.strip() for sec in sample['本文'] for s in sec.get('文', []) if s.strip())
print(f'社会場面: {scene}\n件名: {subject}\n宛先: {receiver}\n本文: {body}')
"
# copy the output and pass it to:
# python pipeline.py --model_dir ./models --email "..."
```

To request the **complete dataset** (1,200 annotated emails), please contact:

**[YOUR_EMAIL_ADDRESS]**

---

## Label Maps

`label_maps.json` contains the label vocabularies used by the classifiers:

| Key | Description |
|---|---|
| `interaction_object_map` | What is being exchanged (e.g., service, information) |
| `interaction_role_map` | Role in the exchange (e.g., giving, requesting) |
| `sender_action_map` | Sender's communicative action (e.g., 依頼, 感謝, 謝罪) |
| `style_map` | Keigo style (e.g., 丁寧語, 謙譲語+丁寧語, インフォーマル) |

---

## Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{liu-etal-2026-evaluating,
  title = {Evaluating Social Intelligence in LLMs via Japanese Honorifics in Email Generation: A Social Semiotic System Perspective},
  author = {Liu, Muxuan and Ishigaki, Tatsuya and Miyao, Yusuke and Takamura, Hiroya and Kobayashi, Ichiro},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {1957--1976},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/54wnt2fwhk8j},
  abstract = {We propose JaSocial, a novel evaluation framework that leverages Japanese emails to comprehensively evaluate large language models’ (LLMs) social intelligence across varied social‑status relationships. The framework integrates three core components. First, we construct and publicly release a meticulously human‑annotated Japanese email dataset covering six distinct social‑status contexts, thereby capturing nuanced shifts in social hierarchy and politeness. Second, we adopt Systemic Functional Linguistics (SFL)—a social-semiotic linguistic theory that explicitly models how linguistic choices realize interpersonal relations and hierarchical distinctions—to classify email content in terms of three perspectives: social relationships, speech functions, and honorific expressions. Based on these perspectives, we design an automated evaluation method that assigns each LLM-generated email a contextual appropriateness score, quantifying how well it reflects socially intelligent behavior. Third, we release the full evaluation code to ensure reproducibility and enable fair cross-model comparisons. JaSocial exposes current LLMs’ limitations in capturing cultural nuance, while providing an open benchmark for future research.}
}
```

## Dataset Access

The dataset is not directly redistributed through this repository.

To request access to the dataset, please contact: **liu.muxuan@is.ocha.ac.jp**


## License
Copyright (c) 2026 Muxuan Liu and contributors.

The dataset is not an open dataset and is not released under a Creative Commons license. Access is granted only upon request.

Use of the dataset is governed by the custom dataset license described in the LICENSE file.

Redistribution of the dataset is not permitted. This includes the original dataset, modified versions, reformatted versions, translated versions, annotated versions, filtered subsets, and any other partial version of the dataset.

If you want to tell others about this dataset, please direct them to this official repository instead of sending them a copy of the dataset.
