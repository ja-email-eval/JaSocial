# classifier.py: 日本語メール敬語タグ分類器（3段階カスケード訓練）
# -*- coding: utf-8 -*-
#
# 訓練 / Training:
#   python classifier.py --mode train_stage1 --data_dir ./data
#   python classifier.py --mode train_stage2 --data_dir ./data
#   python classifier.py --mode train_stage3 --data_dir ./data
#
# 推論 / Inference:
#   python classifier.py --mode inference --input ./email.json --model_dir ./models/stage3_<timestamp>
#
# ラベルマッピング構築 / Build label maps:
#   python classifier.py --mode build_maps --data_dir ./data

import os
import json
import re
import random
import logging
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer, BertModel
from sklearn.model_selection import train_test_split
import datetime
import traceback

#####################################
# 1. 設定ログ、乱数シード及設定パラメータ  #
#####################################

def setup_logger(log_file=None):
    # get current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # set log directory
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # auto-generate log filename based on execution mode
    if log_file is None:
        script_name = os.path.basename(sys.argv[0])
        if "train" in " ".join(sys.argv):
            log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        else:
            log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, log_file)
    
    # remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # configure log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setFormatter(formatter)
    
    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # log script execution info
    script_path = os.path.abspath(sys.argv[0])
    command_line = " ".join(sys.argv)
    logging.info(f"実行スクリプト: {script_path}")
    logging.info(f"実行コマンド: {command_line}")
    logging.info("=== ログシステムの初期化完了 ===")
    logging.info(f"ログファイルパス: {os.path.abspath(log_file)}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"乱数シードを {seed} に設定しました")

def split_data(samples, train_ratio=0.8, val_ratio=0.1):
    """指定割合でサンプルをtrain/val/testに分割。"""
    indices = list(range(len(samples)))
    train_indices, temp_indices = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(1 - val_ratio/(1-train_ratio)), random_state=42
    )
    train_data = [samples[i] for i in train_indices]
    val_data   = [samples[i] for i in val_indices]
    test_data  = [samples[i] for i in test_indices]
    logging.info(f"データセット分割：訓練データ {len(train_data)} 件, 検証データ {len(val_data)} 件, テストデータ {len(test_data)} 件")
    return train_data, val_data, test_data

def create_model_save_dir():
    """
    モデル保存用のディレクトリを作成し、パスを返します。
    ディレクトリ構造：./models/[stage1|stage2|stage3|inference]_YYYYMMDD_HHMMSS/
    """
    base_dir = "./models"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # コマンドライン引数から実行モードを取得
    args = sys.argv
    mode = ""
    for i, arg in enumerate(args):
        if arg == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            break
    
    # モード別のプレフィックスを設定
    prefix = ""
    if mode.startswith("train_"):
        prefix = mode  # train_stage1, train_stage2, train_stage3
    elif mode == "inference":
        prefix = "inference"
    else:
        prefix = "unknown"
    
    # プレフィックスからtrain_を削除
    prefix = prefix.replace("train_", "")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(save_dir)
    logging.info(f"モデル保存ディレクトリを作成しました: {save_dir}")
    return save_dir


def build_label_maps_from_data(data_dir):
    """
    Auto-extract all possible labels from the data directory and build label maps.
    Returns: object_of_exchange_map, role_in_conversation_map, sender_action_map, keigo_type_map
    """
    object_of_exchange = set()
    role_in_conversation = set()
    sender_actions = set()
    keigo_type = set()
    
    # two extra sets for split action labels
    action_types = set()  # action type
    action_contents = set()  # action content
    
    # regex patterns for each label type
    object_pattern = r"第二層:やり取りされるもの:(.*)"
    role_pattern = r"第二層:やり取りにおける役割:(.*)"
    action_pattern = r"第二層:送信者の動き:(.*)"
    style_pattern = r"第三層:(.*)"
    
    # iterate over all JSON files
    file_count = 0
    processed_files = 0
    
    logging.info(f"Scanning directory: {data_dir}")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
                
            file_count += 1
            file_path = os.path.join(root, file)
            
            try:
                logging.info(f"Processing file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
                
                if isinstance(data_json, dict):
                    data_list = [data_json]
                elif isinstance(data_json, list):
                    data_list = data_json
                else:
                    logging.warning(f"File {file_path}: invalid format (not dict or list)")
                    continue
                
                for item_idx, item in enumerate(data_list):
                    sentences, tags_list = extract_sentences_and_tags(item)
                    logging.debug(f"File {file_path} item {item_idx}: {len(sentences)} sentences, {len(tags_list)} tag lists")
                    
                    for sent_idx, (sent, tags) in enumerate(zip(sentences, tags_list)):
                        for tag in tags:
                            if not isinstance(tag, str):
                                continue
                                
                            # extract interaction object
                            object_match = re.search(object_pattern, tag)
                            if object_match:
                                label = object_match.group(1).strip()
                                object_of_exchange.add(label)
                                logging.debug(f"Found interaction object label: {label}")
                                
                            # extract interaction role
                            role_match = re.search(role_pattern, tag)
                            if role_match:
                                label = role_match.group(1).strip()
                                role_in_conversation.add(label)
                                logging.debug(f"Found interaction role label: {label}")
                                
                            # extract sender action
                            action_match = re.search(action_pattern, tag)
                            if action_match:
                                label = action_match.group(1).strip()
                                sender_actions.add(label)
                                logging.debug(f"Found sender action label: {label}")
                                
                                # split action type and content
                                parts = label.split(":")
                                if len(parts) >= 1:
                                    # action type part (before colon)
                                    type_part = parts[0]
                                    for action_type in type_part.split("+"):
                                        action_types.add(action_type.strip())
                                    
                                    # action content part (after colon)
                                    if len(parts) >= 2:
                                        content_part = parts[1]
                                        for action_content in content_part.split("+"):
                                            action_contents.add(action_content.strip())
                                
                            # extract style
                            style_match = re.search(style_pattern, tag)
                            if style_match:
                                label = style_match.group(1).strip()
                                keigo_type.add(label)
                                logging.debug(f"Found style label: {label}")
                
                processed_files += 1
                if processed_files % 10 == 0:
                    logging.info(f"Processed {processed_files}/{file_count} files")
                    
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error in {file_path}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
    
    logging.info(f"Done. Scanned {file_count} files, processed {processed_files} successfully.")
    
    # build mapping dicts
    interaction_object_map = {obj: i for i, obj in enumerate(sorted(object_of_exchange))}
    interaction_role_map = {role: i for i, role in enumerate(sorted(role_in_conversation))}
    sender_action_map = {action: i for i, action in enumerate(sorted(sender_actions))}
    style_map = {style: i for i, style in enumerate(sorted(keigo_type))}
    
    # build split action maps
    action_type_map = {act_type: i for i, act_type in enumerate(sorted(action_types))}
    action_content_map = {act_content: i for i, act_content in enumerate(sorted(action_contents))}
    
    logging.info("Label extraction statistics:")
    logging.info(f"  interaction_object labels: {len(object_of_exchange)}")
    logging.info(f"  interaction_role labels: {len(role_in_conversation)}")
    logging.info(f"  sender_action labels: {len(sender_actions)}")
    logging.info(f"  action_type labels: {len(action_types)}")
    logging.info(f"  action_content labels: {len(action_contents)}")
    logging.info(f"  style labels: {len(keigo_type)}")
    
    # save maps to JSON file
    maps = {
        "interaction_object_map": interaction_object_map,
        "interaction_role_map": interaction_role_map,
        "sender_action_map": sender_action_map,
        "action_type_map": action_type_map,
        "action_content_map": action_content_map,
        "style_map": style_map
    }
    
    maps_file = "./label_maps.json"
    with open(maps_file, "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False, indent=2)

    logging.info(f"Label maps saved to {os.path.abspath(maps_file)}")
    
    return interaction_object_map, interaction_role_map, sender_action_map, action_type_map, action_content_map, style_map

class Config:
    def __init__(self):
        # check if running in inference mode
        mode = None
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]

        # no new directory needed in inference mode
        if mode == "inference":
            self.model_save_dir = ""  # 或者改成 None
        else:
            self.model_save_dir = create_model_save_dir()
        
        # データディレクトリ：メールレベルラベル分フォルダ（例 "同輩", "目下→目上", "目上→目下"）
        self.data_dir = "./data"
        self.pretrained_model = "cl-tohoku/bert-base-japanese-v2"
        self.max_length = 256
        self.batch_size = 16
        self.num_epochs = 5
        self.learning_rate = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Stage1：メールレベルラベルマッピング（必須データ）
        self.social_standing_results_map = {"同輩": 0, "目下→目上": 1, "目上→目下": 2}
        self.role_pair_map = {"学生→友人": 0, "従業員→同僚": 1, "学生→教授": 2,
                              "従業員→上司": 3, "教員→学生": 4, "従業員→部下": 5}
        
        # load Stage2/Stage3 label maps
        self._load_auto_label_maps()
        
    def _load_auto_label_maps(self):
        """Load label maps from the auto-generated mapping file."""
        maps_file = "./label_maps.json"

        if not os.path.exists(maps_file):
            logging.error(f"Label map file not found: {maps_file}")
            raise FileNotFoundError("Label map file not found. Run: python classifier.py --mode build_maps --data_dir ./data")
        
        try:
            logging.info(f"Loading label maps from {maps_file}")
            with open(maps_file, "r", encoding="utf-8") as f:
                maps = json.load(f)
                
            # load all maps
            self.interaction_object_map = maps["interaction_object_map"]
            self.interaction_role_map = maps["interaction_role_map"]
            self.sender_action_map = maps["sender_action_map"]
            self.action_type_map = maps["action_type_map"]
            self.action_content_map = maps["action_content_map"]
            self.style_map = maps["style_map"]
            
            # log loaded label counts
            logging.info("Loaded label maps:")
            logging.info(f"  interaction_object_map: {len(self.interaction_object_map)}")
            logging.info(f"  interaction_role_map: {len(self.interaction_role_map)}")
            logging.info(f"  sender_action_map: {len(self.sender_action_map)}")
            logging.info(f"  action_type_map: {len(self.action_type_map)}")
            logging.info(f"  action_content_map: {len(self.action_content_map)}")
            logging.info(f"  style_map: {len(self.style_map)}")
            
        except Exception as e:
            logging.error(f"Error loading label maps: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to load label maps. Run build_maps mode first.")

#####################################
# 2. データ処理関数                   #
#####################################

def assemble_mail_text(data):
    """
    メールテキストを連結し、社会場面、件名、受信者呼び名及本文を含めます
    """
    social_scene = data.get("社会場面", "")
    subject = data.get("件名", "")
    receiver = data.get("受信者呼び名", "")
    body = ""
    if "本文" in data and isinstance(data["本文"], list):
        body_parts = []
        for section in data["本文"]:
            if isinstance(section, dict) and "文" in section:
                if isinstance(section["文"], list):
                    body_parts.extend(section["文"])
                elif isinstance(section["文"], str):
                    body_parts.append(section["文"])
        body = "\n".join(body_parts)
    combined = f"社会場面: {social_scene}\n件名: {subject}\n宛先: {receiver}\n本文: {body}"
    return combined.strip()

def extract_sentences_and_tags(data):
    """
    JSONから各文テキストと対応するラベルリスト（tag文字列リスト）を抽出。
    戻り値：sentences, tags_list（2つのリスト長さが同じ）
    """
    sentences = []
    tags_list = []
    if "本文" in data and isinstance(data["本文"], list):
        for section in data["本文"]:
            if not isinstance(section, dict):
                continue
            # 文抽出
            sec_sentences = []
            if "文" in section:
                if isinstance(section["文"], list):
                    sec_sentences = section["文"]
                elif isinstance(section["文"], str):
                    sec_sentences = [section["文"]]
            # ラベル抽出
            sec_tags = []
            if "タグ" in section:
                if isinstance(section["タグ"], list):
                    if len(section["タグ"]) == len(sec_sentences) and all(isinstance(x, list) for x in section["タグ"]):
                        sec_tags = section["タグ"]
                    else:
                        sec_tags = [section["タグ"]] * len(sec_sentences)
                else:
                    sec_tags = [[] for _ in sec_sentences]
            else:
                sec_tags = [[] for _ in sec_sentences]
            sentences.extend(sec_sentences)
            tags_list.extend(sec_tags)
    return sentences, tags_list

def parse_second_layer_tags(tags, interaction_object_map, interaction_role_map, sender_action_map):
    """
    第二層ラベル（リスト形式）を多熱ベクトルに解析し、辞書を返します：
      { "interaction_object": [...], "interaction_role": [...], "sender_action": [...] }
    Auto-detect label type and update the corresponding vector.
    """
    # initialize result vectors
    vec = {
        "interaction_object": [0] * len(interaction_object_map),
        "interaction_role": [0] * len(interaction_role_map),
        "sender_action": [0] * len(sender_action_map)
    }
    
    # regex patterns for each label type
    object_pattern = r"第二層:やり取りされるもの:(.*)"
    role_pattern = r"第二層:やり取りにおける役割:(.*)"
    action_pattern = r"第二層:送信者の動き:(.*)"
    
    for tag in tags:
        if not isinstance(tag, str):
            continue
            
        # match interaction object
        object_match = re.search(object_pattern, tag)
        if object_match:
            label_str = object_match.group(1).strip()
            if label_str in interaction_object_map:
                idx = interaction_object_map[label_str]
                vec["interaction_object"][idx] = 1
            else:
                logging.warning(f"Unknown interaction object label: {label_str}")
                
        # match interaction role
        role_match = re.search(role_pattern, tag)
        if role_match:
            label_str = role_match.group(1).strip()
            if label_str in interaction_role_map:
                idx = interaction_role_map[label_str]
                vec["interaction_role"][idx] = 1
            else:
                logging.warning(f"Unknown interaction role label: {label_str}")
                
        # match sender action
        action_match = re.search(action_pattern, tag)
        if action_match:
            label_str = action_match.group(1).strip()
            if label_str in sender_action_map:
                idx = sender_action_map[label_str]
                vec["sender_action"][idx] = 1
            else:
                logging.warning(f"Unknown sender action label: {label_str}")
    
    return vec

def parse_third_layer_tags(tags, style_map):
    """
    第三層ラベルを解析し、多熱ベクトル（リスト）を返します
    """
    vec = [0] * len(style_map)
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if "第三層" in tag:
            parts = tag.split(":")
            if len(parts) >= 2:
                label_str = parts[-1].strip()
                if label_str in style_map:
                    idx = style_map[label_str]
                    vec[idx] = 1
    return vec

#####################################
# 3. データロード関数（自動的にラベルなしデータをフィルタリング）  #
#####################################

def load_data_for_stage1(config):
    logging.info("=== Stage1データの読み込みを開始 ===")
    logging.info(f"データディレクトリ: {config.data_dir}")
    logging.info(f"敬語ラベルマッピング: {config.social_standing_results_map}")
    logging.info(f"役割ペアマッピング: {config.role_pair_map}")
    
    samples = []
    total_files = 0
    processed_files = 0
    skipped_files = 0
    
    # まず総ファイル数を計算
    for folder in os.listdir(config.data_dir):
        folder_path = os.path.join(config.data_dir, folder)
        if os.path.isdir(folder_path):
            total_files += len([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    logging.info(f"総ファイル数: {total_files}")
    
    for folder in os.listdir(config.data_dir):
        if folder not in config.social_standing_results_map:
            logging.warning(f"フォルダ {folder} はsocial_standing_results_mapに含まれていないためスキップします")
            continue
        
        folder_path = os.path.join(config.data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        social_standing_results = config.social_standing_results_map[folder]
        folder_samples = 0
        
        logging.info(f"\nフォルダ処理中: {folder} (social_standing_results: {social_standing_results})")
        
        for file in os.listdir(folder_path):
            if not file.endswith(".json"):
                continue
                
            role_prefix = file.split("_")[0]
            if role_prefix not in config.role_pair_map:
                logging.warning(f"ファイル {file} をスキップ、役割 {role_prefix} がrole_pair_mapに含まれていません")
                skipped_files += 1
                continue
                
            role_label = config.role_pair_map[role_prefix]
            file_path = os.path.join(folder_path, file)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
                    
                file_samples = 0
                if isinstance(data_json, list):
                    for item in data_json:
                        mail_text = assemble_mail_text(item)
                        if mail_text == "":
                            continue
                        samples.append((mail_text, social_standing_results, role_label))
                        file_samples += 1
                elif isinstance(data_json, dict):
                    mail_text = assemble_mail_text(data_json)
                    if mail_text != "":
                        samples.append((mail_text, social_standing_results, role_label))
                        file_samples += 1
                
                folder_samples += file_samples
                processed_files += 1
                logging.info(f"ファイル処理: {file} - {file_samples} 件のサンプルを抽出")
                
            except Exception as e:
                logging.error(f"ファイル {file_path} の処理中にエラーが発生: {str(e)}")
                skipped_files += 1
        
        logging.info(f"フォルダ {folder} の処理完了、{folder_samples} 件のサンプルを抽出")
    
    logging.info("\n=== データ読み込み統計 ===")
    logging.info(f"総ファイル数: {total_files}")
    logging.info(f"処理成功ファイル数: {processed_files}")
    logging.info(f"スキップファイル数: {skipped_files}")
    logging.info(f"総サンプル数: {len(samples)}")
    
    if len(samples) > 0:
        logging.info("\nサンプル例:")
        example = samples[0]
        logging.info(f"テキスト: {example[0][:100]}...")
        logging.info(f"敬語ラベル: {list(config.social_standing_results_map.keys())[example[1]]}")
        logging.info(f"役割ラベル: {list(config.role_pair_map.keys())[example[2]]}")
    
    return samples

def load_data_for_stage2(config):
    """
    第二層ラベルデータを読み込み、サンプルリストを返します：
      [(mail_text, sentence_text, label_dict), ...]
    label_dictには interaction_object、interaction_role、sender_action 多熱ベクトルを含みます。
    もし文に第二層ラベルがない場合はスキップします。
    """
    samples = []
    data_dir = config.data_dir
    logging.info(f"=== Stage2データの読み込みを開始 ===")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
                if isinstance(data_json, dict):
                    data_list = [data_json]
                elif isinstance(data_json, list):
                    data_list = data_json
                else:
                    continue
                for item in data_list:
                    mail_text = assemble_mail_text(item)
                    if mail_text == "":
                        continue
                    sentences, tags_list = extract_sentences_and_tags(item)
                    for s_text, tags in zip(sentences, tags_list):
                        label_dict = parse_second_layer_tags(tags, config.interaction_object_map,
                                                               config.interaction_role_map,
                                                               config.sender_action_map)
                        if sum(label_dict["interaction_object"]) + sum(label_dict["interaction_role"]) + sum(label_dict["sender_action"]) == 0:
                            continue
                        samples.append((mail_text, s_text, label_dict))
                logging.info(f"ファイル {file_path} の読み込みに成功")
            except Exception as e:
                logging.error(f"ファイル {file_path} の読み込み中にエラーが発生: {e}")
    logging.info(f"Stage2 loaded {len(samples)} samples")
    return samples

def load_data_for_stage3(config):
    """
    第三層ラベルデータを読み込み、サンプルリストを返します：
      [(mail_text, sentence_text, style_vector), ...]
    もし文に第三層ラベルがない場合はスキップします。
    """
    samples = []
    data_dir = config.data_dir
    logging.info(f"=== Stage3データの読み込みを開始 ===")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
                if isinstance(data_json, dict):
                    data_list = [data_json]
                elif isinstance(data_json, list):
                    data_list = data_json
                else:
                    continue
                for item in data_list:
                    mail_text = assemble_mail_text(item)
                    if mail_text == "":
                        continue
                    sentences, tags_list = extract_sentences_and_tags(item)
                    for s_text, tags in zip(sentences, tags_list):
                        style_vec = parse_third_layer_tags(tags, config.style_map)
                        if sum(style_vec) == 0:
                            continue
                        samples.append((mail_text, s_text, style_vec))
                logging.info(f"ファイル {file_path} の読み込みに成功")
            except Exception as e:
                logging.error(f"ファイル {file_path} の読み込み中にエラーが発生: {e}")
    logging.info(f"Stage3 loaded {len(samples)} samples")
    return samples

#####################################
# 4. Dataset クラス及 collate 関数         #
#####################################

# Stage1 データセット：メールレベル分類
class Stage1Dataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mail_text, social_standing_results, role_label = self.samples[idx]
        encoding = self.tokenizer(mail_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "social_standing_results": torch.tensor(social_standing_results, dtype=torch.long),
            "role_label": torch.tensor(role_label, dtype=torch.long)
        }

def collate_fn_stage1(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    social_standing_resultss = torch.stack([item["social_standing_results"] for item in batch])
    role_labels = torch.stack([item["role_label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, 
            "social_standing_results": social_standing_resultss, "role_label": role_labels}

# Stage2 データセット：文節級多ラベル予測
class Stage2Dataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mail_text, sent_text, label_dict = self.samples[idx]
        mail_enc = self.tokenizer(mail_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        sent_enc = self.tokenizer(sent_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            "mail_input_ids": mail_enc["input_ids"].squeeze(),
            "mail_attention_mask": mail_enc["attention_mask"].squeeze(),
            "sent_input_ids": sent_enc["input_ids"].squeeze(),
            "sent_attention_mask": sent_enc["attention_mask"].squeeze(),
            "interaction_object": torch.tensor(label_dict["interaction_object"], dtype=torch.float),
            "interaction_role": torch.tensor(label_dict["interaction_role"], dtype=torch.float),
            "sender_action": torch.tensor(label_dict["sender_action"], dtype=torch.float)
        }

def collate_fn_stage2(batch):
    mail_input_ids = torch.stack([item["mail_input_ids"] for item in batch])
    mail_attention_mask = torch.stack([item["mail_attention_mask"] for item in batch])
    sent_input_ids = torch.stack([item["sent_input_ids"] for item in batch])
    sent_attention_mask = torch.stack([item["sent_attention_mask"] for item in batch])
    interaction_object = torch.stack([item["interaction_object"] for item in batch])
    interaction_role = torch.stack([item["interaction_role"] for item in batch])
    sender_action = torch.stack([item["sender_action"] for item in batch])
    return {
        "mail_input_ids": mail_input_ids,
        "mail_attention_mask": mail_attention_mask,
        "sent_input_ids": sent_input_ids,
        "sent_attention_mask": sent_attention_mask,
        "interaction_object": interaction_object,
        "interaction_role": interaction_role,
        "sender_action": sender_action
    }

# Stage3 データセット：文節級 style 分類
class Stage3Dataset(Dataset):
    def __init__(self, samples, tokenizer, max_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mail_text, sent_text, style_vec = self.samples[idx]
        mail_enc = self.tokenizer(mail_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        sent_enc = self.tokenizer(sent_text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        return {
            "mail_input_ids": mail_enc["input_ids"].squeeze(),
            "mail_attention_mask": mail_enc["attention_mask"].squeeze(),
            "sent_input_ids": sent_enc["input_ids"].squeeze(),
            "sent_attention_mask": sent_enc["attention_mask"].squeeze(),
            "style_label": torch.tensor(style_vec, dtype=torch.float)
        }

def collate_fn_stage3(batch):
    mail_input_ids = torch.stack([item["mail_input_ids"] for item in batch])
    mail_attention_mask = torch.stack([item["mail_attention_mask"] for item in batch])
    sent_input_ids = torch.stack([item["sent_input_ids"] for item in batch])
    sent_attention_mask = torch.stack([item["sent_attention_mask"] for item in batch])
    style_labels = torch.stack([item["style_label"] for item in batch])
    return {
        "mail_input_ids": mail_input_ids,
        "mail_attention_mask": mail_attention_mask,
        "sent_input_ids": sent_input_ids,
        "sent_attention_mask": sent_attention_mask,
        "style_label": style_labels
    }

#####################################
# 5. モデル定義                         #
#####################################

# Stage1 Model：メールレベル分類（social_standing_results と role_label を予測）
class Stage1Model(nn.Module):
    def __init__(self, pretrained_model, num_keigo, num_role):
        super(Stage1Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size
        self.keigo_classifier = nn.Linear(hidden_size, num_keigo)
        self.role_classifier = nn.Linear(hidden_size, num_role)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] ベクトル
        keigo_logits = self.keigo_classifier(cls_output)
        role_logits = self.role_classifier(cls_output)
        return keigo_logits, role_logits

# Stage2 Model：文節級多ラベル予測（interaction_object, interaction_role, sender_action を予測）
class Stage2Model(nn.Module):
    def __init__(self, pretrained_model, num_inter_obj, num_inter_role, num_sender_act, num_social_standing_resultss, num_role_labels):
        super(Stage2Model, self).__init__()
        self.bert_mail = BertModel.from_pretrained(pretrained_model)
        self.bert_sent = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert_mail.config.hidden_size
        combined_size = hidden_size * 2 + num_social_standing_resultss + num_role_labels  # Stage1の出力を追加
        self.inter_obj_classifier = nn.Linear(combined_size, num_inter_obj)
        self.inter_role_classifier = nn.Linear(combined_size, num_inter_role)
        self.sender_act_classifier = nn.Linear(combined_size, num_sender_act)
        
    def forward(self, mail_input_ids, mail_attention_mask, sent_input_ids, sent_attention_mask, stage1_social_standing_results, stage1_role_label):
        mail_outputs = self.bert_mail(mail_input_ids, attention_mask=mail_attention_mask)
        sent_outputs = self.bert_sent(sent_input_ids, attention_mask=sent_attention_mask)
        mail_cls = mail_outputs.last_hidden_state[:, 0, :]
        sent_cls = sent_outputs.last_hidden_state[:, 0, :]
        # Stage1の予測ラベルを結合
        combined = torch.cat([mail_cls, sent_cls, stage1_social_standing_results, stage1_role_label], dim=-1)
        inter_obj_logits = self.inter_obj_classifier(combined)
        inter_role_logits = self.inter_role_classifier(combined)
        sender_act_logits = self.sender_act_classifier(combined)
        return inter_obj_logits, inter_role_logits, sender_act_logits

# Stage3 Model：文節級 style 分類（style ラベルを予測）
class Stage3Model(nn.Module):
    def __init__(self, pretrained_model, num_keigo_type, num_inter_obj, num_inter_role, num_sender_act):
        super(Stage3Model, self).__init__()
        self.bert_mail = BertModel.from_pretrained(pretrained_model)
        self.bert_sent = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert_mail.config.hidden_size
        # Stage2の出力を追加
        combined_size = hidden_size * 2 + num_inter_obj + num_inter_role + num_sender_act
        self.classifier = nn.Linear(combined_size, num_keigo_type)
        
    def forward(self, mail_input_ids, mail_attention_mask, sent_input_ids, sent_attention_mask, 
                stage2_inter_obj, stage2_inter_role, stage2_sender_act):
        mail_outputs = self.bert_mail(mail_input_ids, attention_mask=mail_attention_mask)
        sent_outputs = self.bert_sent(sent_input_ids, attention_mask=sent_attention_mask)
        mail_cls = mail_outputs.last_hidden_state[:, 0, :]
        sent_cls = sent_outputs.last_hidden_state[:, 0, :]
        # Stage2の予測結果を結合
        combined = torch.cat([mail_cls, sent_cls, stage2_inter_obj, stage2_inter_role, stage2_sender_act], dim=-1)
        logits = self.classifier(combined)
        return logits

#####################################
# 6. 訓練関数                         #
#####################################

def train_stage1(config):
    # let setup_logger auto-generate the log filename
    setup_logger()
    logging.info("\n=== Stage1モデル（メールレベル分類）の訓練開始 ===")
    logging.info(f"デバイス: {config.device}")
    logging.info(f"事前学習モデル: {config.pretrained_model}")
    logging.info(f"最大長: {config.max_length}")
    logging.info(f"バッチサイズ: {config.batch_size}")
    logging.info(f"学習率: {config.learning_rate}")
    logging.info(f"エポック数: {config.num_epochs}")
    
    tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)
    logging.info("トークナイザーの読み込み完了")
    
    samples = load_data_for_stage1(config)
    train_samples, val_samples, test_samples = split_data(samples)
    
    train_dataset = Stage1Dataset(train_samples, tokenizer, config.max_length)
    val_dataset = Stage1Dataset(val_samples, tokenizer, config.max_length)
    test_dataset = Stage1Dataset(test_samples, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_stage1)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage1)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage1)
    
    model = Stage1Model(config.pretrained_model, num_keigo=len(config.social_standing_results_map), num_role=len(config.role_pair_map))
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    logging.info("\nモデルパラメータ統計:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"総パラメータ数: {total_params:,}")
    logging.info(f"学習可能パラメータ数: {trainable_params:,}")
    
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        correct_keigo = 0
        correct_role = 0
        total = 0
        
        logging.info(f"\n=== エポック {epoch+1}/{config.num_epochs} ===")
        
        # 訓練フェーズ
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            social_standing_resultss = batch["social_standing_results"].to(config.device)
            role_labels = batch["role_label"].to(config.device)
            
            keigo_logits, role_logits = model(input_ids, attention_mask)
            loss = criterion(keigo_logits, social_standing_resultss) + criterion(role_logits, role_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 訓練accuracy calculation
            pred_keigo = torch.argmax(keigo_logits, dim=1)
            pred_role = torch.argmax(role_logits, dim=1)
            correct_keigo += (pred_keigo == social_standing_resultss).sum().item()
            correct_role += (pred_role == role_labels).sum().item()
            total += social_standing_resultss.size(0)
            
            if (i+1) % 10 == 0:
                avg_loss = total_loss / (i+1)
                keigo_acc = correct_keigo / total * 100
                role_acc = correct_role / total * 100
                logging.info(f"バッチ {i+1}/{len(train_loader)} - 損失: {avg_loss:.4f}, "
                           f"敬語精度: {keigo_acc:.2f}%, 役割精度: {role_acc:.2f}%")
        
        # 訓練フェーズの統計
        epoch_loss = total_loss / len(train_loader)
        epoch_keigo_acc = correct_keigo / total * 100
        epoch_role_acc = correct_role / total * 100
        logging.info(f"\nエポック {epoch+1} 訓練フェーズ統計:")
        logging.info(f"平均損失: {epoch_loss:.4f}")
        logging.info(f"敬語精度: {epoch_keigo_acc:.2f}%")
        logging.info(f"役割精度: {epoch_role_acc:.2f}%")
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        correct_keigo = 0
        correct_role = 0
        total = 0
        
        logging.info("\n検証開始...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                social_standing_resultss = batch["social_standing_results"].to(config.device)
                role_labels = batch["role_label"].to(config.device)
                
                keigo_logits, role_logits = model(input_ids, attention_mask)
                loss = criterion(keigo_logits, social_standing_resultss) + criterion(role_logits, role_labels)
                val_loss += loss.item()
                
                pred_keigo = torch.argmax(keigo_logits, dim=1)
                pred_role = torch.argmax(role_logits, dim=1)
                correct_keigo += (pred_keigo == social_standing_resultss).sum().item()
                correct_role += (pred_role == role_labels).sum().item()
                total += social_standing_resultss.size(0)
        
        # 検証フェーズの統計
        val_loss = val_loss / len(val_loader)
        val_keigo_acc = correct_keigo / total * 100
        val_role_acc = correct_role / total * 100
        val_acc = (correct_keigo + correct_role) / (2 * total) * 100
        
        logging.info(f"\nエポック {epoch+1} 検証フェーズ統計:")
        logging.info(f"検証損失: {val_loss:.4f}")
        logging.info(f"検証敬語精度: {val_keigo_acc:.2f}%")
        logging.info(f"検証役割精度: {val_role_acc:.2f}%")
        logging.info(f"検証総合精度: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, "stage1_model.bin"))
            logging.info(f"モデルを保存しました！新しい最高検証精度: {best_val_acc:.2f}%")
    
    # 全エポック終了後、テストデータでの評価
    logging.info("\n=== 全エポック終了後のテストデータ評価 ===")
    model.load_state_dict(torch.load(os.path.join(config.model_save_dir, "stage1_model.bin")))
    model.eval()
    test_loss = 0
    correct_keigo = 0
    correct_role = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            social_standing_resultss = batch["social_standing_results"].to(config.device)
            role_labels = batch["role_label"].to(config.device)
            
            keigo_logits, role_logits = model(input_ids, attention_mask)
            loss = criterion(keigo_logits, social_standing_resultss) + criterion(role_logits, role_labels)
            test_loss += loss.item()
            
            pred_keigo = torch.argmax(keigo_logits, dim=1)
            pred_role = torch.argmax(role_logits, dim=1)
            correct_keigo += (pred_keigo == social_standing_resultss).sum().item()
            correct_role += (pred_role == role_labels).sum().item()
            total += social_standing_resultss.size(0)
    
    # テストフェーズの統計
    test_loss = test_loss / len(test_loader)
    test_keigo_acc = correct_keigo / total * 100
    test_role_acc = correct_role / total * 100
    test_acc = (correct_keigo + correct_role) / (2 * total) * 100
    
    logging.info(f"テスト損失: {test_loss:.4f}")
    logging.info(f"テスト敬語精度: {test_keigo_acc:.2f}%")
    logging.info(f"テスト役割精度: {test_role_acc:.2f}%")
    logging.info(f"テスト総合精度: {test_acc:.2f}%")
    
    logging.info("\n=== 訓練完了 ===")
    logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
    logging.info(f"テスト総合精度: {test_acc:.2f}%")

def compute_multilabel_accuracy(preds, labels, threshold=0.5):
    """
    Calculate element-wise accuracy for multi-label classification.
    """
    preds_bin = (preds >= threshold).float()
    correct = (preds_bin == labels).float().sum().item()
    total = labels.numel()
    return correct / total

def compute_multilabel_accuracy_detailed(preds, labels, label_map, threshold=0.5):
    """
    Calculate per-label accuracy for multi-label classification.
    """
    preds_bin = (preds >= threshold).float()
    per_label_correct = {}
    per_label_total = {}
    per_label_acc = {}
    
    # initialize counters
    for label in label_map:
        per_label_correct[label] = 0
        per_label_total[label] = 0
    
    # count correct predictions per label
    for i, label_name in enumerate(label_map):
        idx = label_map[label_name]
        correct = (preds_bin[:, idx] == labels[:, idx]).float().sum().item()
        total = labels.size(0)
        per_label_correct[label_name] = correct
        per_label_total[label_name] = total
        per_label_acc[label_name] = correct / total * 100 if total > 0 else 0
    
    # compute overall accuracy
    correct = (preds_bin == labels).float().sum().item()
    total = labels.numel()
    overall_acc = correct / total * 100 if total > 0 else 0
    
    return overall_acc, per_label_acc

def train_stage2(config):
    setup_logger()
    # print all label map info first
    logging.info("\n===== Label map details =====")
    
    # print interaction object map
    logging.info("\nInteraction object map (interaction_object_map):")
    for label, idx in sorted(config.interaction_object_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # print interaction role map
    logging.info("\nInteraction role map (interaction_role_map):")
    for label, idx in sorted(config.interaction_role_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # print sender action map
    logging.info("\nSender action map (sender_action_map):")
    for label, idx in sorted(config.sender_action_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # also print action type/content maps if present
    if hasattr(config, 'action_type_map'):
        logging.info("\nAction type map (action_type_map):")
        for label, idx in sorted(config.action_type_map.items(), key=lambda x: x[1]):
            logging.info(f"  {idx}: {label}")
    
    if hasattr(config, 'action_content_map'):
        logging.info("\nAction content map (action_content_map):")
        for label, idx in sorted(config.action_content_map.items(), key=lambda x: x[1]):
            logging.info(f"  {idx}: {label}")
    
    logging.info("\n===== Label map print complete =====")
    
    logging.info("\n=== Stage2モデル（文節級多ラベル予測）の訓練開始 ===")
    
    # setup_logger already called above, do not call again
    # training code starts here
    
    tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)
    
    # Stage1モデルの読み込み
    stage1_model = Stage1Model(config.pretrained_model, num_keigo=len(config.social_standing_results_map), num_role=len(config.role_pair_map))
    latest_stage1_model = sorted([d for d in os.listdir("./models") if d.startswith("stage1_")])[-1]
    stage1_model.load_state_dict(torch.load(os.path.join("./models", latest_stage1_model, "stage1_model.bin"), map_location=config.device))
    stage1_model.to(config.device)
    stage1_model.eval()
    
    samples = load_data_for_stage2(config)
    train_samples, val_samples, test_samples = split_data(samples)
    
    train_dataset = Stage2Dataset(train_samples, tokenizer, config.max_length)
    val_dataset = Stage2Dataset(val_samples, tokenizer, config.max_length)
    test_dataset = Stage2Dataset(test_samples, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_stage2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage2)
    
    model = Stage2Model(config.pretrained_model, 
                       num_inter_obj=len(config.interaction_object_map),
                       num_inter_role=len(config.interaction_role_map),
                       num_sender_act=len(config.sender_action_map),
                       num_social_standing_resultss=len(config.social_standing_results_map),
                       num_role_labels=len(config.role_pair_map))
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_inter_obj_acc = 0
        total_inter_role_acc = 0
        total_sender_act_acc = 0
        total_batches = 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            mail_input_ids = batch["mail_input_ids"].to(config.device)
            mail_attention_mask = batch["mail_attention_mask"].to(config.device)
            sent_input_ids = batch["sent_input_ids"].to(config.device)
            sent_attention_mask = batch["sent_attention_mask"].to(config.device)
            
            # Stage1の予測を取得
            with torch.no_grad():
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
            
            inter_obj_logits, inter_role_logits, sender_act_logits = model(
                mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                stage1_social_standing_results, stage1_role_label
            )
            
            loss = criterion(inter_obj_logits, batch["interaction_object"].to(config.device)) + \
                   criterion(inter_role_logits, batch["interaction_role"].to(config.device)) + \
                   criterion(sender_act_logits, batch["sender_action"].to(config.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 訓練時のaccuracy calculation
            inter_obj_acc = compute_multilabel_accuracy(torch.sigmoid(inter_obj_logits), batch["interaction_object"].to(config.device))
            inter_role_acc = compute_multilabel_accuracy(torch.sigmoid(inter_role_logits), batch["interaction_role"].to(config.device))
            sender_act_acc = compute_multilabel_accuracy(torch.sigmoid(sender_act_logits), batch["sender_action"].to(config.device))
            
            total_inter_obj_acc += inter_obj_acc
            total_inter_role_acc += inter_role_acc
            total_sender_act_acc += sender_act_acc
            total_batches += 1
            
            if (i+1) % 10 == 0:
                avg_loss = total_loss / (i+1)
                avg_inter_obj_acc = total_inter_obj_acc / total_batches * 100
                avg_inter_role_acc = total_inter_role_acc / total_batches * 100
                avg_sender_act_acc = total_sender_act_acc / total_batches * 100
                
                logging.info(f"Stage2 Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}:")
                logging.info(f"  Loss = {avg_loss:.4f}")
                logging.info(f"  やり取りされるもの精度: {avg_inter_obj_acc:.2f}%")
                logging.info(f"  やり取りにおける役割精度: {avg_inter_role_acc:.2f}%")
                logging.info(f"  送信者の動き精度: {avg_sender_act_acc:.2f}%")
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_inter_obj_acc = 0
        val_inter_role_acc = 0
        val_sender_act_acc = 0
        val_batches = 0
        
        with torch.no_grad():
            # collect all predictions and labels
            all_inter_obj_preds = []
            all_inter_role_preds = []
            all_sender_act_preds = []
            all_inter_obj_labels = []
            all_inter_role_labels = []
            all_sender_act_labels = []
            
            for batch in val_loader:
                mail_input_ids = batch["mail_input_ids"].to(config.device)
                mail_attention_mask = batch["mail_attention_mask"].to(config.device)
                sent_input_ids = batch["sent_input_ids"].to(config.device)
                sent_attention_mask = batch["sent_attention_mask"].to(config.device)
                
                # Stage1の予測を取得
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
                
                inter_obj_logits, inter_role_logits, sender_act_logits = model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                
                loss = criterion(inter_obj_logits, batch["interaction_object"].to(config.device)) + \
                       criterion(inter_role_logits, batch["interaction_role"].to(config.device)) + \
                       criterion(sender_act_logits, batch["sender_action"].to(config.device))
                val_loss += loss.item()
                
                # accuracy calculation
                inter_obj_acc = compute_multilabel_accuracy(torch.sigmoid(inter_obj_logits), batch["interaction_object"].to(config.device))
                inter_role_acc = compute_multilabel_accuracy(torch.sigmoid(inter_role_logits), batch["interaction_role"].to(config.device))
                sender_act_acc = compute_multilabel_accuracy(torch.sigmoid(sender_act_logits), batch["sender_action"].to(config.device))
                
                val_inter_obj_acc += inter_obj_acc
                val_inter_role_acc += inter_role_acc
                val_sender_act_acc += sender_act_acc
                val_batches += 1
                
                all_inter_obj_preds.append(torch.sigmoid(inter_obj_logits).cpu())
                all_inter_role_preds.append(torch.sigmoid(inter_role_logits).cpu())
                all_sender_act_preds.append(torch.sigmoid(sender_act_logits).cpu())
                all_inter_obj_labels.append(batch["interaction_object"].cpu())
                all_inter_role_labels.append(batch["interaction_role"].cpu())
                all_sender_act_labels.append(batch["sender_action"].cpu())
            
            # merge all batch predictions and labels
            all_inter_obj_preds = torch.cat(all_inter_obj_preds, dim=0)
            all_inter_role_preds = torch.cat(all_inter_role_preds, dim=0)
            all_sender_act_preds = torch.cat(all_sender_act_preds, dim=0)
            all_inter_obj_labels = torch.cat(all_inter_obj_labels, dim=0)
            all_inter_role_labels = torch.cat(all_inter_role_labels, dim=0)
            all_sender_act_labels = torch.cat(all_sender_act_labels, dim=0)
            
            # compute per-label accuracy
            obj_acc, obj_per_label_acc = compute_multilabel_accuracy_detailed(
                all_inter_obj_preds, all_inter_obj_labels, config.interaction_object_map)
            role_acc, role_per_label_acc = compute_multilabel_accuracy_detailed(
                all_inter_role_preds, all_inter_role_labels, config.interaction_role_map)
            act_acc, act_per_label_acc = compute_multilabel_accuracy_detailed(
                all_sender_act_preds, all_sender_act_labels, config.sender_action_map)
            
            # detailed accuracy report
            logging.info("\n===== Detailed label accuracy report =====")
            
            logging.info("\nInteraction object label accuracy:")
            for label, acc in sorted(obj_per_label_acc.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {label}: {acc:.2f}%")
            
            logging.info("\nInteraction role label accuracy:")
            for label, acc in sorted(role_per_label_acc.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {label}: {acc:.2f}%")
            
            logging.info("\nSender action label accuracy:")
            # show top-20 and bottom-5 to keep output concise
            sorted_act_accs = sorted(act_per_label_acc.items(), key=lambda x: x[1], reverse=True)
            logging.info("  Top 20 labels by accuracy:")
            for label, acc in sorted_act_accs[:20]:
                logging.info(f"  {label}: {acc:.2f}%")
            
            if len(sorted_act_accs) > 20:
                logging.info("\n  Bottom 5 labels by accuracy:")
                for label, acc in sorted_act_accs[-5:]:
                    logging.info(f"  {label}: {acc:.2f}%")
            
            # show predictions for a few random samples
            logging.info("\n===== Sample prediction examples =====")
            try:
                sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
                
                for idx in sample_indices:
                    try:
                        sample = val_dataset[idx]
                        logging.info(f"\nSample #{idx}:")
                        logging.info(f"Sample keys: {list(sample.keys())}")
                        
                        # decode email content
                        mail_ids = sample["mail_input_ids"].tolist()
                        mail_ids_clean = [tid for tid in mail_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                        if mail_ids_clean:
                            mail_text = tokenizer.decode(mail_ids_clean)
                            logging.info(f"Email content: {mail_text[:100]}..." if len(mail_text) > 100 else mail_text)
                        else:
                            logging.info("Unable to decode email content")
                        
                        # decode sentence content
                        sent_ids = sample["sent_input_ids"].tolist()
                        sent_ids_clean = [tid for tid in sent_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                        if sent_ids_clean:
                            sent_text = tokenizer.decode(sent_ids_clean)
                            logging.info(f"Sentence: {sent_text}")
                        else:
                            logging.info("Unable to decode sentence content")
                        
                        # run model prediction on tokenized input
                        mail_input_ids = sample["mail_input_ids"].unsqueeze(0).to(config.device)
                        mail_attention_mask = sample["mail_attention_mask"].unsqueeze(0).to(config.device)
                        sent_input_ids = sample["sent_input_ids"].unsqueeze(0).to(config.device)
                        sent_attention_mask = sample["sent_attention_mask"].unsqueeze(0).to(config.device)
                        
                        with torch.no_grad():
                            keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                            stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                            stage1_role_label = torch.softmax(role_logits, dim=1)
                            
                            inter_obj_logits, inter_role_logits, sender_act_logits = model(
                                mail_input_ids, mail_attention_mask,
                                sent_input_ids, sent_attention_mask,
                                stage1_social_standing_results, stage1_role_label
                            )
                        
                        # get prediction probabilities
                        obj_probs = torch.sigmoid(inter_obj_logits)[0].cpu().numpy()
                        role_probs = torch.sigmoid(inter_role_logits)[0].cpu().numpy()
                        act_probs = torch.sigmoid(sender_act_logits)[0].cpu().numpy()
                        
                        # show ground truth labels
                        logging.info("Ground truth:")
                        
                        logging.info("  Interaction object:")
                        true_obj_indices = np.where(sample["interaction_object"].numpy() == 1)[0]
                        for i in true_obj_indices:
                            label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        logging.info("  Interaction role:")
                        true_role_indices = np.where(sample["interaction_role"].numpy() == 1)[0]
                        for i in true_role_indices:
                            label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        logging.info("  Sender action:")
                        true_act_indices = np.where(sample["sender_action"].numpy() == 1)[0]
                        for i in true_act_indices:
                            label = [k for k, v in config.sender_action_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        # show predictions
                        logging.info("\nPredictions:")
                        
                        # show interaction object predictions
                        logging.info("  Interaction object (top 3):")
                        top_obj_indices = np.argsort(obj_probs)[::-1][:3]
                        for i in top_obj_indices:
                            label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                            is_true = i in true_obj_indices
                            logging.info(f"    - {label}: {obj_probs[i]:.4f} {'✓' if is_true else '✗'}")
                        
                        # show interaction role predictions
                        logging.info("  Interaction role (top 3):")
                        top_role_indices = np.argsort(role_probs)[::-1][:3]
                        for i in top_role_indices:
                            label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                            is_true = i in true_role_indices
                            logging.info(f"    - {label}: {role_probs[i]:.4f} {'✓' if is_true else '✗'}")
                        
                        # show sender action predictions
                        logging.info("  Sender action (top 5):")
                        top_act_indices = np.argsort(act_probs)[::-1][:5]
                        for i in top_act_indices:
                            label = [k for k, v in config.sender_action_map.items() if v == i][0]
                            is_true = i in true_act_indices
                            logging.info(f"    - {label}: {act_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    except Exception as e:
                        logging.error(f"Error processing sample #{idx}: {str(e)}")
                        logging.error(traceback.format_exc())
            except Exception as e:
                logging.error(f"Sample prediction error: {str(e)}")
                logging.error(traceback.format_exc())
            
            logging.info("===== Detail report end =====")
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_inter_obj_acc = val_inter_obj_acc / val_batches * 100
        avg_val_inter_role_acc = val_inter_role_acc / val_batches * 100
        avg_val_sender_act_acc = val_sender_act_acc / val_batches * 100
        avg_val_acc = (avg_val_inter_obj_acc + avg_val_inter_role_acc + avg_val_sender_act_acc) / 3
        
        logging.info(f"\nStage2 Epoch {epoch+1} 検証結果:")
        logging.info(f"  Val Loss = {avg_val_loss:.4f}")
        logging.info(f"  やり取りされるもの精度: {avg_val_inter_obj_acc:.2f}%")
        logging.info(f"  やり取りにおける役割精度: {avg_val_inter_role_acc:.2f}%")
        logging.info(f"  送信者の動き精度: {avg_val_sender_act_acc:.2f}%")
        logging.info(f"  平均精度: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, "stage2_model.bin"))
            logging.info(f"モデルを保存しました！新しい最高検証精度: {best_val_acc:.2f}%")
        
        logging.info("\n=== Stage2 training complete ===")
        logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
        
        # detailed per-label report after validation
        logging.info("\n===== Per-label accuracy report =====")
        
        # collect predictions and labels for all validation data
        model.eval()
        all_obj_preds = []
        all_role_preds = []
        all_act_preds = []
        all_obj_labels = []
        all_role_labels = []
        all_act_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                mail_input_ids = batch["mail_input_ids"].to(config.device)
                mail_attention_mask = batch["mail_attention_mask"].to(config.device)
                sent_input_ids = batch["sent_input_ids"].to(config.device)
                sent_attention_mask = batch["sent_attention_mask"].to(config.device)
                
                # Stage1 prediction
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
                
                # Stage2 prediction
                inter_obj_logits, inter_role_logits, sender_act_logits = model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                
                obj_probs = torch.sigmoid(inter_obj_logits)
                role_probs = torch.sigmoid(inter_role_logits)
                act_probs = torch.sigmoid(sender_act_logits)
                
                all_obj_preds.append(obj_probs.cpu())
                all_role_preds.append(role_probs.cpu())
                all_act_preds.append(act_probs.cpu())
                all_obj_labels.append(batch["interaction_object"].cpu())
                all_role_labels.append(batch["interaction_role"].cpu())
                all_act_labels.append(batch["sender_action"].cpu())
            
            # merge all batch predictions and labels
            all_obj_preds = torch.cat(all_obj_preds, dim=0)
            all_role_preds = torch.cat(all_role_preds, dim=0)
            all_act_preds = torch.cat(all_act_preds, dim=0)
            all_obj_labels = torch.cat(all_obj_labels, dim=0)
            all_role_labels = torch.cat(all_role_labels, dim=0)
            all_act_labels = torch.cat(all_act_labels, dim=0)
        
        # compute accuracy per label
        # interaction object label accuracy
        logging.info("\nInteraction object label accuracy details:")
        for label, idx in config.interaction_object_map.items():
            true_pos = ((all_obj_preds[:, idx] >= 0.5) & (all_obj_labels[:, idx] == 1)).sum().item()
            true_neg = ((all_obj_preds[:, idx] < 0.5) & (all_obj_labels[:, idx] == 0)).sum().item()
            false_pos = ((all_obj_preds[:, idx] >= 0.5) & (all_obj_labels[:, idx] == 0)).sum().item()
            false_neg = ((all_obj_preds[:, idx] < 0.5) & (all_obj_labels[:, idx] == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = (all_obj_labels[:, idx] == 1).sum().item()
            
            logging.info(f"  {label}: precision={precision*100:.2f}%, recall={recall*100:.2f}%, F1={f1*100:.2f}%, support={support}")
        
        # interaction role label accuracy
        logging.info("\nInteraction role label accuracy details:")
        for label, idx in config.interaction_role_map.items():
            true_pos = ((all_role_preds[:, idx] >= 0.5) & (all_role_labels[:, idx] == 1)).sum().item()
            true_neg = ((all_role_preds[:, idx] < 0.5) & (all_role_labels[:, idx] == 0)).sum().item()
            false_pos = ((all_role_preds[:, idx] >= 0.5) & (all_role_labels[:, idx] == 0)).sum().item()
            false_neg = ((all_role_preds[:, idx] < 0.5) & (all_role_labels[:, idx] == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = (all_role_labels[:, idx] == 1).sum().item()
            
            logging.info(f"  {label}: precision={precision*100:.2f}%, recall={recall*100:.2f}%, F1={f1*100:.2f}%, support={support}")
        
        # sender action label accuracy (top 20 by support count)
        logging.info("\nSender action label accuracy details (top 20 by support):")
        act_metrics = []
        for label, idx in config.sender_action_map.items():
            true_pos = ((all_act_preds[:, idx] >= 0.5) & (all_act_labels[:, idx] == 1)).sum().item()
            true_neg = ((all_act_preds[:, idx] < 0.5) & (all_act_labels[:, idx] == 0)).sum().item()
            false_pos = ((all_act_preds[:, idx] >= 0.5) & (all_act_labels[:, idx] == 0)).sum().item()
            false_neg = ((all_act_preds[:, idx] < 0.5) & (all_act_labels[:, idx] == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = (all_act_labels[:, idx] == 1).sum().item()
            
            act_metrics.append((label, precision, recall, f1, support))
        
        # 按支持数排序
        act_metrics.sort(key=lambda x: x[4], reverse=True)
        for label, precision, recall, f1, support in act_metrics[:20]:
            logging.info(f"  {label}: precision={precision*100:.2f}%, recall={recall*100:.2f}%, F1={f1*100:.2f}%, support={support}")
        
        # show a few sample predictions
        logging.info("\n===== Sample prediction examples =====")
        try:
            sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
            
            for idx in sample_indices:
                try:
                    sample = val_dataset[idx]
                    logging.info(f"\nSample #{idx}:")
                    logging.info(f"Sample keys: {list(sample.keys())}")
                    
                    # decode email content
                    mail_ids = sample["mail_input_ids"].tolist()
                    mail_ids_clean = [tid for tid in mail_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                    if mail_ids_clean:
                        mail_text = tokenizer.decode(mail_ids_clean)
                        logging.info(f"Email content: {mail_text[:100]}..." if len(mail_text) > 100 else mail_text)
                    else:
                        logging.info("Unable to decode email content")
                    
                    # decode sentence content
                    sent_ids = sample["sent_input_ids"].tolist()
                    sent_ids_clean = [tid for tid in sent_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                    if sent_ids_clean:
                        sent_text = tokenizer.decode(sent_ids_clean)
                        logging.info(f"Sentence: {sent_text}")
                    else:
                        logging.info("Unable to decode sentence content")
                    
                    # run model prediction on tokenized input
                    mail_input_ids = sample["mail_input_ids"].unsqueeze(0).to(config.device)
                    mail_attention_mask = sample["mail_attention_mask"].unsqueeze(0).to(config.device)
                    sent_input_ids = sample["sent_input_ids"].unsqueeze(0).to(config.device)
                    sent_attention_mask = sample["sent_attention_mask"].unsqueeze(0).to(config.device)
                    
                    with torch.no_grad():
                        keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                        stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                        stage1_role_label = torch.softmax(role_logits, dim=1)
                        
                        inter_obj_logits, inter_role_logits, sender_act_logits = model(
                            mail_input_ids, mail_attention_mask,
                            sent_input_ids, sent_attention_mask,
                            stage1_social_standing_results, stage1_role_label
                        )
                        
                        obj_probs = torch.sigmoid(inter_obj_logits)[0].cpu().numpy()
                        role_probs = torch.sigmoid(inter_role_logits)[0].cpu().numpy()
                        act_probs = torch.sigmoid(sender_act_logits)[0].cpu().numpy()
                    
                    # show ground truth labels
                    logging.info("Ground truth:")
                    
                    logging.info("  Interaction object:")
                    true_obj_indices = np.where(sample["interaction_object"].numpy() == 1)[0]
                    for i in true_obj_indices:
                        label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    logging.info("  Interaction role:")
                    true_role_indices = np.where(sample["interaction_role"].numpy() == 1)[0]
                    for i in true_role_indices:
                        label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    logging.info("  Sender action:")
                    true_act_indices = np.where(sample["sender_action"].numpy() == 1)[0]
                    for i in true_act_indices:
                        label = [k for k, v in config.sender_action_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    # show predictions
                    logging.info("\nPredictions:")
                    
                    logging.info("  Interaction object (top 3):")
                    top_obj_indices = np.argsort(obj_probs)[::-1][:3]
                    for i in top_obj_indices:
                        label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                        is_true = i in true_obj_indices
                        logging.info(f"    - {label}: {obj_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    logging.info("  Interaction role (top 3):")
                    top_role_indices = np.argsort(role_probs)[::-1][:3]
                    for i in top_role_indices:
                        label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                        is_true = i in true_role_indices
                        logging.info(f"    - {label}: {role_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    logging.info("  Sender action (top 5):")
                    top_act_indices = np.argsort(act_probs)[::-1][:5]
                    for i in top_act_indices:
                        label = [k for k, v in config.sender_action_map.items() if v == i][0]
                        is_true = i in true_act_indices
                        logging.info(f"    - {label}: {act_probs[i]:.4f} {'✓' if is_true else '✗'}")
                
                except Exception as e:
                    logging.error(f"Error processing sample #{idx}: {str(e)}")
                    logging.error(traceback.format_exc())
        except Exception as e:
            logging.error(f"Sample prediction error: {str(e)}")
            logging.error(traceback.format_exc())
        
        logging.info("\n===== Detail report end =====")

    # 全エポック終了後、テストデータでの評価
    logging.info("\n=== 全エポック終了後のテストデータ評価 ===")
    model.load_state_dict(torch.load(os.path.join(config.model_save_dir, "stage2_model.bin")))
    model.eval()
    test_loss = 0
    test_inter_obj_acc = 0
    test_inter_role_acc = 0
    test_sender_act_acc = 0
    test_batches = 0
    
    with torch.no_grad():
        # collect test set predictions and labels
        all_inter_obj_preds = []
        all_inter_role_preds = []
        all_sender_act_preds = []
        all_inter_obj_labels = []
        all_inter_role_labels = []
        all_sender_act_labels = []
        
        for batch in test_loader:
            mail_input_ids = batch["mail_input_ids"].to(config.device)
            mail_attention_mask = batch["mail_attention_mask"].to(config.device)
            sent_input_ids = batch["sent_input_ids"].to(config.device)
            sent_attention_mask = batch["sent_attention_mask"].to(config.device)
            
            # Stage1の予測を取得
            keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
            stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
            stage1_role_label = torch.softmax(role_logits, dim=1)
            
            inter_obj_logits, inter_role_logits, sender_act_logits = model(
                mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                stage1_social_standing_results, stage1_role_label
            )
            
            loss = criterion(inter_obj_logits, batch["interaction_object"].to(config.device)) + \
                   criterion(inter_role_logits, batch["interaction_role"].to(config.device)) + \
                   criterion(sender_act_logits, batch["sender_action"].to(config.device))
            test_loss += loss.item()
            
            # accuracy calculation
            inter_obj_acc = compute_multilabel_accuracy(torch.sigmoid(inter_obj_logits), batch["interaction_object"].to(config.device))
            inter_role_acc = compute_multilabel_accuracy(torch.sigmoid(inter_role_logits), batch["interaction_role"].to(config.device))
            sender_act_acc = compute_multilabel_accuracy(torch.sigmoid(sender_act_logits), batch["sender_action"].to(config.device))
            
            test_inter_obj_acc += inter_obj_acc
            test_inter_role_acc += inter_role_acc
            test_sender_act_acc += sender_act_acc
            test_batches += 1
            
            all_inter_obj_preds.append(torch.sigmoid(inter_obj_logits).cpu())
            all_inter_role_preds.append(torch.sigmoid(inter_role_logits).cpu())
            all_sender_act_preds.append(torch.sigmoid(sender_act_logits).cpu())
            all_inter_obj_labels.append(batch["interaction_object"].cpu())
            all_inter_role_labels.append(batch["interaction_role"].cpu())
            all_sender_act_labels.append(batch["sender_action"].cpu())
        
        # merge all batch predictions and labels
        all_inter_obj_preds = torch.cat(all_inter_obj_preds, dim=0)
        all_inter_role_preds = torch.cat(all_inter_role_preds, dim=0)
        all_sender_act_preds = torch.cat(all_sender_act_preds, dim=0)
        all_inter_obj_labels = torch.cat(all_inter_obj_labels, dim=0)
        all_inter_role_labels = torch.cat(all_inter_role_labels, dim=0)
        all_sender_act_labels = torch.cat(all_sender_act_labels, dim=0)
        
        # compute per-label accuracy
        obj_acc, obj_per_label_acc = compute_multilabel_accuracy_detailed(
            all_inter_obj_preds, all_inter_obj_labels, config.interaction_object_map)
        role_acc, role_per_label_acc = compute_multilabel_accuracy_detailed(
            all_inter_role_preds, all_inter_role_labels, config.interaction_role_map)
        act_acc, act_per_label_acc = compute_multilabel_accuracy_detailed(
            all_sender_act_preds, all_sender_act_labels, config.sender_action_map)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_inter_obj_acc = test_inter_obj_acc / test_batches * 100
    avg_test_inter_role_acc = test_inter_role_acc / test_batches * 100
    avg_test_sender_act_acc = test_sender_act_acc / test_batches * 100
    avg_test_acc = (avg_test_inter_obj_acc + avg_test_inter_role_acc + avg_test_sender_act_acc) / 3
    
    logging.info(f"\nStage2 テスト結果:")
    logging.info(f"  Test Loss = {avg_test_loss:.4f}")
    logging.info(f"  やり取りされるもの精度: {avg_test_inter_obj_acc:.2f}%")
    logging.info(f"  やり取りにおける役割精度: {avg_test_inter_role_acc:.2f}%")
    logging.info(f"  送信者の動き精度: {avg_test_sender_act_acc:.2f}%")
    logging.info(f"  平均精度: {avg_test_acc:.2f}%")
    
    # detailed accuracy report
    logging.info(f"\n===== Test set per-label accuracy report =====")
    
    logging.info("\nInteraction object label accuracy:")
    for label, acc in sorted(obj_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\nInteraction role label accuracy:")
    for label, acc in sorted(role_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\nSender action label accuracy:")
    # show top-20 and bottom-5 to keep output concise
    sorted_act_accs = sorted(act_per_label_acc.items(), key=lambda x: x[1], reverse=True)
    logging.info("  Top 20 labels by accuracy:")
    for label, acc in sorted_act_accs[:20]:
        logging.info(f"  {label}: {acc:.2f}%")
    
    if len(sorted_act_accs) > 20:
        logging.info("\n  Bottom 5 labels by accuracy:")
        for label, acc in sorted_act_accs[-5:]:
            logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n=== Stage2 training complete ===")
    logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
    logging.info(f"テスト総合精度: {avg_test_acc:.2f}%")

def train_stage3(config):
    # print all label maps first
    logging.info("===== Stage3 training start =====")
    logging.info("Label map contents:")
    
    # print style map
    logging.info("\nStyle map (style_map):")
    for label, idx in sorted(config.style_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    logging.info("\n===== Label map print complete =====")
    
    setup_logger()
    logging.info("\n=== Stage3モデル（文節級 style 分類）の訓練開始 ===")

    tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)
    
    # Stage1とStage2モデルの読み込み
    stage1_model = Stage1Model(config.pretrained_model, num_keigo=len(config.social_standing_results_map), num_role=len(config.role_pair_map))
    stage2_model = Stage2Model(config.pretrained_model,
                             num_inter_obj=len(config.interaction_object_map),
                             num_inter_role=len(config.interaction_role_map),
                             num_sender_act=len(config.sender_action_map),
                             num_social_standing_resultss=len(config.social_standing_results_map),
                             num_role_labels=len(config.role_pair_map))
    
    # 最新のモデルを読み込む
    latest_stage1_model = sorted([d for d in os.listdir("./models") if d.startswith("stage1_")])[-1]
    latest_stage2_model = sorted([d for d in os.listdir("./models") if d.startswith("stage2_")])[-1]

    stage1_model.load_state_dict(torch.load(os.path.join("./models", latest_stage1_model, "stage1_model.bin"), map_location=config.device))
    stage2_model.load_state_dict(torch.load(os.path.join("./models", latest_stage2_model, "stage2_model.bin"), map_location=config.device))
    
    stage1_model.to(config.device)
    stage2_model.to(config.device)
    stage1_model.eval()
    stage2_model.eval()
    
    samples = load_data_for_stage3(config)
    train_samples, val_samples, test_samples = split_data(samples)
    
    train_dataset = Stage3Dataset(train_samples, tokenizer, config.max_length)
    val_dataset = Stage3Dataset(val_samples, tokenizer, config.max_length)
    test_dataset = Stage3Dataset(test_samples, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_stage3)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage3)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_stage3)
    
    model = Stage3Model(config.pretrained_model,
                       num_keigo_type=len(config.style_map),
                       num_inter_obj=len(config.interaction_object_map),
                       num_inter_role=len(config.interaction_role_map),
                       num_sender_act=len(config.sender_action_map))
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_style_accs = torch.zeros(len(config.style_map), dtype=torch.float)
        total_batches = 0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            mail_input_ids = batch["mail_input_ids"].to(config.device)
            mail_attention_mask = batch["mail_attention_mask"].to(config.device)
            sent_input_ids = batch["sent_input_ids"].to(config.device)
            sent_attention_mask = batch["sent_attention_mask"].to(config.device)
            
            # Stage1とStage2の予測を取得
            with torch.no_grad():
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
                
                inter_obj_logits, inter_role_logits, sender_act_logits = stage2_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                stage2_inter_obj = torch.sigmoid(inter_obj_logits)
                stage2_inter_role = torch.sigmoid(inter_role_logits)
                stage2_sender_act = torch.sigmoid(sender_act_logits)
            
            logits = model(mail_input_ids, mail_attention_mask,
                         sent_input_ids, sent_attention_mask,
                         stage2_inter_obj, stage2_inter_role, stage2_sender_act)
            
            loss = criterion(logits, batch["style_label"].to(config.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 各スタイルの精度を計算
            probs = torch.sigmoid(logits)
            style_labels = batch["style_label"].to(config.device)
            for j in range(len(config.style_map)):
                acc = compute_multilabel_accuracy(probs[:, j:j+1], style_labels[:, j:j+1])
                total_style_accs[j] += acc
            total_batches += 1
            
            if (i+1) % 10 == 0:
                avg_loss = total_loss / (i+1)
                avg_style_accs = total_style_accs / total_batches * 100
                
                logging.info(f"Stage3 Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}:")
                logging.info(f"  Loss = {avg_loss:.4f}")
                for style_name, acc in zip(config.style_map.keys(), avg_style_accs):
                    logging.info(f"  {style_name}精度: {acc:.2f}%")
        
        # 検証フェーズ
        model.eval()
        val_loss = 0
        val_style_accs = torch.zeros(len(config.style_map), dtype=torch.float)
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                mail_input_ids = batch["mail_input_ids"].to(config.device)
                mail_attention_mask = batch["mail_attention_mask"].to(config.device)
                sent_input_ids = batch["sent_input_ids"].to(config.device)
                sent_attention_mask = batch["sent_attention_mask"].to(config.device)
                
                # Stage1とStage2の予測を取得
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
                
                inter_obj_logits, inter_role_logits, sender_act_logits = stage2_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                stage2_inter_obj = torch.sigmoid(inter_obj_logits)
                stage2_inter_role = torch.sigmoid(inter_role_logits)
                stage2_sender_act = torch.sigmoid(sender_act_logits)
                
                logits = model(mail_input_ids, mail_attention_mask,
                             sent_input_ids, sent_attention_mask,
                             stage2_inter_obj, stage2_inter_role, stage2_sender_act)
                
                loss = criterion(logits, batch["style_label"].to(config.device))
                val_loss += loss.item()
                
                # 各スタイルの精度を計算
                probs = torch.sigmoid(logits)
                style_labels = batch["style_label"].to(config.device)
                for j in range(len(config.style_map)):
                    acc = compute_multilabel_accuracy(probs[:, j:j+1], style_labels[:, j:j+1])
                    val_style_accs[j] += acc
                val_batches += 1
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_style_accs = val_style_accs / val_batches * 100
        avg_val_acc = avg_val_style_accs.mean().item()
        
        logging.info(f"\nStage3 Epoch {epoch+1} 検証結果:")
        logging.info(f"  Val Loss = {avg_val_loss:.4f}")
        for style_name, acc in zip(config.style_map.keys(), avg_val_style_accs):
            logging.info(f"  {style_name}精度: {acc:.2f}%")
        logging.info(f"  平均精度: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, "stage3_model.bin"))
            logging.info(f"モデルを保存しました！新しい最高検証精度: {best_val_acc:.2f}%")

    # 全エポック終了後、テストデータでの評価
    logging.info("\n=== 全エポック終了後のテストデータ評価 ===")
    model.load_state_dict(torch.load(os.path.join(config.model_save_dir, "stage3_model.bin")))
    model.eval()
    test_loss = 0
    test_style_acc = 0
    test_batches = 0
    
    with torch.no_grad():
        # collect test set predictions and labels
        all_style_preds = []
        all_style_labels = []
        
        for batch in test_loader:
            mail_input_ids = batch["mail_input_ids"].to(config.device)
            mail_attention_mask = batch["mail_attention_mask"].to(config.device)
            sent_input_ids = batch["sent_input_ids"].to(config.device)
            sent_attention_mask = batch["sent_attention_mask"].to(config.device)
            
            # Stage1の予測を取得
            keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
            stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
            stage1_role_label = torch.softmax(role_logits, dim=1)
            
            # Stage2の予測を取得
            inter_obj_logits, inter_role_logits, sender_act_logits = stage2_model(
                mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                stage1_social_standing_results, stage1_role_label
            )
            stage2_inter_obj = torch.sigmoid(inter_obj_logits)
            stage2_inter_role = torch.sigmoid(inter_role_logits)
            stage2_sender_act = torch.sigmoid(sender_act_logits)
            
            # Stage3の予測
            style_logits = model(
                mail_input_ids, mail_attention_mask,
                sent_input_ids, sent_attention_mask,
                stage2_inter_obj, stage2_inter_role, stage2_sender_act
            )
            
            loss = criterion(style_logits, batch["style_label"].to(config.device))
            test_loss += loss.item()
            
            # accuracy calculation
            style_acc = compute_multilabel_accuracy(torch.sigmoid(style_logits), batch["style_label"].to(config.device))
            test_style_acc += style_acc
            test_batches += 1
            
            all_style_preds.append(torch.sigmoid(style_logits).cpu())
            all_style_labels.append(batch["style_label"].cpu())
        
        # merge all batch predictions and labels
        all_style_preds = torch.cat(all_style_preds, dim=0)
        all_style_labels = torch.cat(all_style_labels, dim=0)
        
        # compute per-label accuracy
        style_acc, style_per_label_acc = compute_multilabel_accuracy_detailed(
            all_style_preds, all_style_labels, config.style_map)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_style_acc = test_style_acc / test_batches * 100
    
    logging.info(f"\nStage3 テスト結果:")
    logging.info(f"  Test Loss = {avg_test_loss:.4f}")
    logging.info(f"  スタイル精度: {avg_test_style_acc:.2f}%")
    
    # detailed accuracy report
    logging.info(f"\n===== Test set per-label accuracy report =====")
    
    logging.info("\nStyle label accuracy:")
    for label, acc in sorted(style_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n=== Stage3 training complete ===")
    logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
    logging.info(f"テスト総合精度: {avg_test_style_acc:.2f}%")

#####################################
# 7. 推論（Inference）関数            #
#####################################

def inference_pipeline(config, input_json_path, model_dir):
    """
    Inference pipeline: process an input JSON file and generate keigo predictions.
    
    参数:
      config: Config object
      input_json_path: path to the input JSON file
      model_dir: directory containing all three stage model files
    """
    setup_logger()
    print(f"\n===== 分類開始 =====")
    print(f"Input file: {input_json_path}")
    print(f"Model directory: {model_dir}")
    
    # all three models are in the same directory with different filenames
    stage1_model_path = os.path.join(model_dir, "stage1_model.bin")
    stage2_model_path = os.path.join(model_dir, "stage2_model.bin")
    stage3_model_path = os.path.join(model_dir, "stage3_model.bin")
    
    # check that all model files exist
    for model_path in [stage1_model_path, stage2_model_path, stage3_model_path]:
        if not os.path.exists(model_path):
            print(f"Error: model file not found: {model_path}")
            return
    
    try:
        # load tokenizer
        print("Loading tokenizer...")
        tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)
        
        # load Stage1 model
        print("Loading Stage1 model...")
        stage1_model = Stage1Model(config.pretrained_model, 
                                  num_keigo=len(config.social_standing_results_map), 
                                  num_role=len(config.role_pair_map))
        stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=config.device))
        stage1_model.to(config.device)
        stage1_model.eval()
        
        # load Stage2 model
        print("Loading Stage2 model...")
        stage2_model = Stage2Model(config.pretrained_model,
                                  num_inter_obj=len(config.interaction_object_map),
                                  num_inter_role=len(config.interaction_role_map),
                                  num_sender_act=len(config.sender_action_map),
                                  num_social_standing_resultss=len(config.social_standing_results_map),
                                  num_role_labels=len(config.role_pair_map))
        stage2_model.load_state_dict(torch.load(stage2_model_path, map_location=config.device))
        stage2_model.to(config.device)
        stage2_model.eval()
        
        # load Stage3 model
        print("Loading Stage3 model...")
        stage3_model = Stage3Model(config.pretrained_model,
                                  num_keigo_type=len(config.style_map),
                                  num_inter_obj=len(config.interaction_object_map),
                                  num_inter_role=len(config.interaction_role_map),
                                  num_sender_act=len(config.sender_action_map))
        stage3_model.load_state_dict(torch.load(stage3_model_path, map_location=config.device))
        stage3_model.to(config.device)
        stage3_model.eval()
        
        # read input JSON
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # extract email text
        mail_text = ""
        if "mail_text" in data:
            mail_text = data["mail_text"]
        elif "本文" in data:
            # try extracting text from 本文 field
            if isinstance(data["本文"], list):
                mail_parts = []
                for section in data["本文"]:
                    if isinstance(section, dict) and "文" in section:
                        if isinstance(section["文"], list):
                            mail_parts.extend(section["文"])
                        else:
                            mail_parts.append(section["文"])
                mail_text = "\n".join(mail_parts)
            else:
                mail_text = str(data["本文"])
        
        # fallback: use assemble_mail_text
        if not mail_text and "assemble_mail_text" in globals():
            mail_text = assemble_mail_text(data)
        
        if not mail_text:
            print("Error: could not extract email text from input JSON")
            return
        
        # extract sentences
        sentences = []
        if "sentences" in data:
            sentences = data["sentences"]
        elif "本文" in data and isinstance(data["本文"], list):
            for section in data["本文"]:
                if isinstance(section, dict) and "文" in section:
                    if isinstance(section["文"], list):
                        sentences.extend(section["文"])
                    else:
                        sentences.append(section["文"])
        
        if not sentences:
            # fallback: treat whole email as one sentence
            sentences = [mail_text]
        
        # print email content
        print(f"\n===== メール内容 =====")
        print(mail_text)
        print(f"\n{len(sentences)} 個の文を分析する必要があります")
        
        # Stage1 prediction for the whole email
        mail_enc = tokenizer(mail_text, truncation=True, padding='max_length', 
                            max_length=config.max_length, return_tensors="pt")
        mail_input_ids = mail_enc["input_ids"].to(config.device)
        mail_attention_mask = mail_enc["attention_mask"].to(config.device)
        
        with torch.no_grad():
            # Stage1 prediction (whole email)
            keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
            keigo_probs = torch.softmax(keigo_logits, dim=1)[0].cpu().numpy()
            role_probs = torch.softmax(role_logits, dim=1)[0].cpu().numpy()
            
            # get predicted label indices
            keigo_pred = np.argmax(keigo_probs)
            role_pred = np.argmax(role_probs)
            
            # get label names
            social_standing_results = [k for k, v in config.social_standing_results_map.items() if v == keigo_pred][0]
            role_label = [k for k, v in config.role_pair_map.items() if v == role_pred][0]
            
            # print Stage1 predictions
            print(f"\n===== メール全体レベル予測結果 =====")
            print(f"社会関係: {social_standing_results} (確率: {keigo_probs[keigo_pred]:.4f})")
            print(f"役割関係: {role_label} (確率: {role_probs[role_pred]:.4f})")
            
            # convert Stage1 output to tensors for next stages
            stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
            stage1_role_label = torch.softmax(role_logits, dim=1)
        
        # process each sentence
        results = []
        for i, sentence in enumerate(sentences):
            print(f"\n===== 処理する文 {i+1}/{len(sentences)} =====")
            print(f"文の内容: {sentence}")
            
            # encode sentence
            sent_enc = tokenizer(sentence, truncation=True, padding='max_length', 
                                max_length=config.max_length, return_tensors="pt")
            sent_input_ids = sent_enc["input_ids"].to(config.device)
            sent_attention_mask = sent_enc["attention_mask"].to(config.device)
            
            with torch.no_grad():
                # Stage2 prediction (using Stage1 output)
                inter_obj_logits, inter_role_logits, sender_act_logits = stage2_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                
                obj_probs = torch.sigmoid(inter_obj_logits)[0].cpu().numpy()
                role_probs = torch.sigmoid(inter_role_logits)[0].cpu().numpy()
                act_probs = torch.sigmoid(sender_act_logits)[0].cpu().numpy()
                
                # Stage3 prediction
                stage2_inter_obj = torch.sigmoid(inter_obj_logits)
                stage2_inter_role = torch.sigmoid(inter_role_logits)
                stage2_sender_act = torch.sigmoid(sender_act_logits)
                
                style_logits = stage3_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage2_inter_obj, stage2_inter_role, stage2_sender_act
                )
                
                style_probs = torch.sigmoid(style_logits)[0].cpu().numpy()
            
            # top-3 interaction object predictions
            top_objects = []
            top_obj_indices = np.argsort(obj_probs)[::-1][:3]
            for idx in top_obj_indices:
                label = [k for k, v in config.interaction_object_map.items() if v == idx][0]
                top_objects.append({"label": label, "probability": float(obj_probs[idx])})
            
            # top-3 interaction role predictions
            top_roles = []
            top_role_indices = np.argsort(role_probs)[::-1][:3]
            for idx in top_role_indices:
                label = [k for k, v in config.interaction_role_map.items() if v == idx][0]
                top_roles.append({"label": label, "probability": float(role_probs[idx])})
            
            # top-5 sender action predictions
            top_actions = []
            top_act_indices = np.argsort(act_probs)[::-1][:5]
            for idx in top_act_indices:
                label = [k for k, v in config.sender_action_map.items() if v == idx][0]
                top_actions.append({"label": label, "probability": float(act_probs[idx])})
            
            # get style label predictions
            keigo_type_results = []
            for idx, (style_name, _) in enumerate(config.style_map.items()):
                keigo_type_results.append({"style": style_name, "probability": float(style_probs[idx])})
            
            # print results
            print("\n===== 文レベル予測結果 =====")
            
            print("\nやり取りされるもの:")
            for obj in top_objects:
                print(f"  - {obj['label']}: {obj['probability']:.4f}")
            
            print("\nやり取りにおける役割:")
            for role in top_roles:
                print(f"  - {role['label']}: {role['probability']:.4f}")
            
            print("\n送信者の動き:")
            for act in top_actions:
                print(f"  - {act['label']}: {act['probability']:.4f}")
            
            print("\n敬語表現:")
            for style in keigo_type_results:
                print(f"  - {style['style']}: {style['probability']:.4f}")
            
            # append to results
            sentence_result = {
                "sentence": sentence,
                "social_standing": social_standing_results,
                "role_relationship": role_label,
                "object_of_exchange": top_objects,
                "role_in_conversation": top_roles,
                "sender_actions": top_actions,
                "keigo_type": keigo_type_results
            }
            results.append(sentence_result)
        
        # build final result object
        final_result = {
            "mail_text": mail_text,
            "mail_level_prediction": {
                "keigo_type": social_standing_results,
                "role_relation": role_label
            },
            "sentence_results": results
        }
        
        # save result to JSON
        output_path = input_json_path.replace(".json", "_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\nClassification complete. Results saved to {output_path}")
        
        # print summary
        print("\n===== まとめ =====")
        print(f"メールには {len(sentences)} 個の文が含まれています")
        print(f"メールレベル予測: 敬語タイプ = {social_standing_results}, 役割関係 = {role_label}")
        
        return final_result
        
    except Exception as e:
        print(f"分類過程でエラーが発生しました: {str(e)}")
        print(traceback.format_exc())
        return None

#####################################
# 8. メイン関数（argparseによるモード選択）  #
#####################################

def main():
    parser = argparse.ArgumentParser(description="敬語パイプライン学習と推論")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train_stage1", "train_stage2", "train_stage3", "inference", "build_maps"],
                        help="モード選択：train_stage1, train_stage2, train_stage3, inference, build_maps")
    parser.add_argument("--input", type=str, default="", help="推論モード時の入力JSONファイルパス")
    parser.add_argument("--model_dir", type=str, default="", help="推論モード時のモデルディレクトリ")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="訓練データが格納されているディレクトリ (デフォルト: ./data)")
    parser.add_argument("--auto_maps", action="store_true", help="Auto-build label maps from data")
    
    args = parser.parse_args()
    
    set_seed(42)
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir

    if args.mode == "build_maps":
        setup_logger()
        logging.info("Building label maps...")
        _ = build_label_maps_from_data(config.data_dir)
        return
    
    if args.mode == "train_stage1":
        train_stage1(config)
    elif args.mode == "train_stage2":
        train_stage2(config)
    elif args.mode == "train_stage3":
        train_stage3(config)
    elif args.mode == "inference":
        if not args.input:
            print("Error: --input is required for inference mode")
            return
        if not args.model_dir:
            print("Error: --model_dir is required for inference mode")
            return

        inference_pipeline(config, args.input, args.model_dir)

if __name__ == "__main__":
    main()
