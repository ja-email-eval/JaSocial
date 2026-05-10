"""
Keigo Scoring Pipeline
======================
このスクリプトは入力された日本語メールの敬語使用状況を3段階モデルで総合採点します。

3-stage classification models:
  Stage1: メールレベルの社会関係・役割関係認識
  Stage2: 文レベルのやり取り対象・役割・送信者行動認識
  Stage3: 文レベルの敬語スタイル分類

使い方 / Usage:
  python pipeline.py --email "メール本文..." [--model_dir ./models] [--label_maps ./label_maps.json]
  python pipeline.py --email "社会場面: 学生→教授\n件名: 質問\n本文: お世話になっております。"

出力 / Output:
  採点結果は ./runs/ に JSON 形式で保存されます。
  Scoring results are saved as JSON files in ./runs/.

モデルファイルの配置 / Model file placement:
  models/
    stage1_<timestamp>/stage1_model.bin
    stage2_<timestamp>/stage2_model.bin
    stage3_<timestamp>/stage3_model.bin
  モデルは Google Drive からダウンロードしてください（README 参照）。
"""

import os
import json
import re
import logging
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel
import datetime
import traceback

# ==========================================
# 1. Configuration
# ==========================================

class PipelineConfig:
    def __init__(self, model_root="./models", label_maps_path="./label_maps.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = "cl-tohoku/bert-base-japanese-v2"
        self.max_length = 256
        self.model_root = model_root

        self.social_standing_results_map = {"同輩": 0, "目下→目上": 1, "目上→目下": 2}
        self.role_pair_map = {"学生→友人": 0, "従業員→同僚": 1, "学生→教授": 2,
                              "従業員→上司": 3, "教員→学生": 4, "従業員→部下": 5}

        self._load_label_maps(label_maps_path)

    def _load_label_maps(self, maps_file):
        if not os.path.exists(maps_file):
            raise FileNotFoundError(
                f"Label map file not found: {maps_file}\n"
                f"label_maps.json must be in the same directory as this script."
            )
        with open(maps_file, "r", encoding="utf-8") as f:
            maps = json.load(f)
        self.interaction_object_map = maps["interaction_object_map"]
        self.interaction_role_map = maps["interaction_role_map"]
        self.sender_action_map = maps["sender_action_map"]
        self.style_map = maps["style_map"]

# ==========================================
# 2. Model Definitions (3-Stage BERT Classifiers)
# ==========================================

class Stage1Model(nn.Module):
    def __init__(self, pretrained_model, num_keigo, num_role):
        super(Stage1Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.keigo_classifier = nn.Linear(hidden_size, num_keigo)
        self.role_classifier = nn.Linear(hidden_size, num_role)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.keigo_classifier(cls_output), self.role_classifier(cls_output)

class Stage2Model(nn.Module):
    def __init__(self, pretrained_model, num_inter_obj, num_inter_role, num_sender_act,
                 num_social_standing, num_role_labels):
        super(Stage2Model, self).__init__()
        self.bert_mail = BertModel.from_pretrained(pretrained_model)
        self.bert_sent = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert_mail.config.hidden_size
        combined_size = hidden_size * 2 + num_social_standing + num_role_labels
        self.inter_obj_classifier = nn.Linear(combined_size, num_inter_obj)
        self.inter_role_classifier = nn.Linear(combined_size, num_inter_role)
        self.sender_act_classifier = nn.Linear(combined_size, num_sender_act)

    def forward(self, mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                stage1_social, stage1_role):
        m_cls = self.bert_mail(mail_input_ids, attention_mask=mail_attention_mask).last_hidden_state[:, 0, :]
        s_cls = self.bert_sent(sent_input_ids, attention_mask=sent_attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat([m_cls, s_cls, stage1_social, stage1_role], dim=-1)
        return (self.inter_obj_classifier(combined),
                self.inter_role_classifier(combined),
                self.sender_act_classifier(combined))

class Stage3Model(nn.Module):
    def __init__(self, pretrained_model, num_keigo_type,
                 num_inter_obj, num_inter_role, num_sender_act):
        super(Stage3Model, self).__init__()
        self.bert_mail = BertModel.from_pretrained(pretrained_model)
        self.bert_sent = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert_mail.config.hidden_size
        combined_size = hidden_size * 2 + num_inter_obj + num_inter_role + num_sender_act
        self.classifier = nn.Linear(combined_size, num_keigo_type)

    def forward(self, mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                s2_obj, s2_role, s2_act):
        m_cls = self.bert_mail(mail_input_ids, attention_mask=mail_attention_mask).last_hidden_state[:, 0, :]
        s_cls = self.bert_sent(sent_input_ids, attention_mask=sent_attention_mask).last_hidden_state[:, 0, :]
        combined = torch.cat([m_cls, s_cls, s2_obj, s2_role, s2_act], dim=-1)
        return self.classifier(combined)

# ==========================================
# 3. Inference & Scoring
# ==========================================

def run_scoring_pipeline(email_text, config, output_dir="./runs", reference_labels=None):
    os.makedirs(output_dir, exist_ok=True)

    def get_latest_model(stage_name):
        model_dirs = sorted([d for d in os.listdir(config.model_root)
                             if d.startswith(f"{stage_name}_")])
        if not model_dirs:
            raise FileNotFoundError(
                f"{stage_name} のモデルが見つかりません。\n"
                f"Model directory '{config.model_root}' に "
                f"'{stage_name}_<timestamp>/{stage_name}_model.bin' を配置してください。\n"
                f"Google Drive からダウンロード → README 参照。"
            )
        return os.path.join(config.model_root, model_dirs[-1], f"{stage_name}_model.bin")

    tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)

    s1_path = get_latest_model("stage1")
    s2_path = get_latest_model("stage2")
    s3_path = get_latest_model("stage3")

    s1_model = Stage1Model(
        config.pretrained_model,
        len(config.social_standing_results_map),
        len(config.role_pair_map)
    ).to(config.device)
    s1_model.load_state_dict(torch.load(s1_path, map_location=config.device))
    s1_model.eval()

    s2_model = Stage2Model(
        config.pretrained_model,
        len(config.interaction_object_map),
        len(config.interaction_role_map),
        len(config.sender_action_map),
        len(config.social_standing_results_map),
        len(config.role_pair_map)
    ).to(config.device)
    s2_model.load_state_dict(torch.load(s2_path, map_location=config.device))
    s2_model.eval()

    s3_model = Stage3Model(
        config.pretrained_model,
        len(config.style_map),
        len(config.interaction_object_map),
        len(config.interaction_role_map),
        len(config.sender_action_map)
    ).to(config.device)
    s3_model.load_state_dict(torch.load(s3_path, map_location=config.device))
    s3_model.eval()

    sentences = re.split(r'(?<=[。？！])', email_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    mail_enc = tokenizer(
        email_text, truncation=True, padding='max_length',
        max_length=config.max_length, return_tensors="pt"
    ).to(config.device)

    with torch.no_grad():
        s1_social_logits, s1_role_logits = s1_model(
            mail_enc["input_ids"], mail_enc["attention_mask"]
        )
        s1_social_soft = torch.softmax(s1_social_logits, dim=1)
        s1_role_soft = torch.softmax(s1_role_logits, dim=1)

        social_pred = list(config.social_standing_results_map.keys())[
            torch.argmax(s1_social_soft).item()
        ]
        role_pred = list(config.role_pair_map.keys())[
            torch.argmax(s1_role_soft).item()
        ]

    sentence_results = []
    for sent in sentences:
        sent_enc = tokenizer(
            sent, truncation=True, padding='max_length',
            max_length=config.max_length, return_tensors="pt"
        ).to(config.device)
        with torch.no_grad():
            s2_obj_logits, s2_role_logits, s2_act_logits = s2_model(
                mail_enc["input_ids"], mail_enc["attention_mask"],
                sent_enc["input_ids"], sent_enc["attention_mask"],
                s1_social_soft, s1_role_soft
            )
            s2_obj_sig = torch.sigmoid(s2_obj_logits)
            s2_role_sig = torch.sigmoid(s2_role_logits)
            s2_act_sig = torch.sigmoid(s2_act_logits)

            s3_logits = s3_model(
                mail_enc["input_ids"], mail_enc["attention_mask"],
                sent_enc["input_ids"], sent_enc["attention_mask"],
                s2_obj_sig, s2_role_sig, s2_act_sig
            )
            s3_sig = torch.sigmoid(s3_logits).cpu().numpy()[0]

            def get_top_k(probs, label_map, k=3):
                indices = np.argsort(probs)[::-1][:k]
                return [
                    {"label": list(label_map.keys())[i], "score": float(probs[i])}
                    for i in indices if probs[i] > 0.1
                ]

            sentence_results.append({
                "text": sent,
                "objects": get_top_k(s2_obj_sig.cpu().numpy()[0], config.interaction_object_map),
                "roles": get_top_k(s2_role_sig.cpu().numpy()[0], config.interaction_role_map),
                "actions": get_top_k(s2_act_sig.cpu().numpy()[0], config.sender_action_map, k=5),
                "styles": get_top_k(s3_sig, config.style_map, k=5)
            })

    s1_score = (torch.max(s1_social_soft).item() + torch.max(s1_role_soft).item()) / 2
    if sentence_results:
        total = 0
        for res in sentence_results:
            obj_score   = res["objects"][0]["score"]  if res["objects"]  else 0
            role_score  = res["roles"][0]["score"]    if res["roles"]    else 0
            act_score   = res["actions"][0]["score"]  if res["actions"]  else 0
            style_score = res["styles"][0]["score"]   if res["styles"]   else 0
            total += (obj_score + role_score + act_score + style_score) / 4
        avg_s2_s3 = total / len(sentence_results)
    else:
        avg_s2_s3 = 0

    final_score = (s1_score + avg_s2_s3) / 2

    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_email": email_text,
        "final_score": round(final_score, 4),
        "email_level": {
            "social_standing": social_pred,
            "role_relation": role_pred,
            "confidence": round(s1_score, 4)
        },
        "sentence_level": sentence_results
    }
    if reference_labels:
        output["reference_labels"] = reference_labels

    filename = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_path = os.path.join(output_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Scoring complete. Final Score: {round(final_score, 4)}")
    print(f"Results saved to: {save_path}")
    return output

# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="日本語メールの敬語使用を3段階BERTモデルで採点するパイプライン"
    )
    parser.add_argument(
        "--email", type=str, required=True,
        help="採点するメールのテキスト。社会場面プレフィックスを含めると精度が上がります。"
             " 例: '社会場面: 学生→教授\\n件名: 質問\\n本文: ...'"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./models",
        help="モデルファイルが格納されているルートディレクトリ (デフォルト: ./models)"
    )
    parser.add_argument(
        "--label_maps", type=str, default="./label_maps.json",
        help="ラベルマッピング JSON ファイルのパス (デフォルト: ./label_maps.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./runs",
        help="採点結果 JSON の保存先ディレクトリ (デフォルト: ./runs)"
    )
    args = parser.parse_args()

    try:
        config = PipelineConfig(
            model_root=args.model_dir,
            label_maps_path=args.label_maps
        )
        run_scoring_pipeline(
            email_text=args.email,
            config=config,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] {e}")
        print(traceback.format_exc())
        sys.exit(1)
