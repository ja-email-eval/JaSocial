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
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置日志目录
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 如果没有指定日志文件名，根据当前执行的脚本类型生成文件名
    if log_file is None:
        script_name = os.path.basename(sys.argv[0])
        if "train" in " ".join(sys.argv):
            log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        else:
            log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, log_file)
    
    # 删除所有既存のハンドラ
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 設定ログフォーマット
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # ファイルハンドラ
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    file_handler.setFormatter(formatter)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 設定根ログレコーダー
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # 记录执行的脚本信息
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
    从数据目录中自动提取所有可能的标签并构建映射
    返回: object_of_exchange_map, role_in_conversation_map, sender_action_map, keigo_type_map
    """
    object_of_exchange = set()
    role_in_conversation = set()
    sender_actions = set()
    keigo_type = set()
    
    # 新增两个集合来存储拆分后的标签
    action_types = set()  # 行为类型
    action_contents = set()  # 行为内容
    
    # 定义标签类型的正则表达式模式
    object_pattern = r"第二層:やり取りされるもの:(.*)"
    role_pattern = r"第二層:やり取りにおける役割:(.*)"
    action_pattern = r"第二層:送信者の動き:(.*)"
    style_pattern = r"第三層:(.*)"
    
    # 遍历所有JSON文件
    file_count = 0
    processed_files = 0
    
    logging.info(f"开始扫描目录: {data_dir}")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
                
            file_count += 1
            file_path = os.path.join(root, file)
            
            try:
                logging.info(f"处理文件: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
                
                if isinstance(data_json, dict):
                    data_list = [data_json]
                elif isinstance(data_json, list):
                    data_list = data_json
                else:
                    logging.warning(f"文件 {file_path} 格式不正确，既不是字典也不是列表")
                    continue
                
                for item_idx, item in enumerate(data_list):
                    sentences, tags_list = extract_sentences_and_tags(item)
                    logging.debug(f"文件 {file_path} 中的项目 {item_idx} 包含 {len(sentences)} 个句子和 {len(tags_list)} 个标签列表")
                    
                    for sent_idx, (sent, tags) in enumerate(zip(sentences, tags_list)):
                        for tag in tags:
                            if not isinstance(tag, str):
                                continue
                                
                            # 提取交互对象
                            object_match = re.search(object_pattern, tag)
                            if object_match:
                                label = object_match.group(1).strip()
                                object_of_exchange.add(label)
                                logging.debug(f"找到交互对象标签: {label}")
                                
                            # 提取交互角色
                            role_match = re.search(role_pattern, tag)
                            if role_match:
                                label = role_match.group(1).strip()
                                role_in_conversation.add(label)
                                logging.debug(f"找到交互角色标签: {label}")
                                
                            # 提取发送者行为
                            action_match = re.search(action_pattern, tag)
                            if action_match:
                                label = action_match.group(1).strip()
                                sender_actions.add(label)
                                logging.debug(f"找到发送者行为标签: {label}")
                                
                                # 拆分行为类型和内容
                                parts = label.split(":")
                                if len(parts) >= 1:
                                    # 处理行为类型部分 (冒号前)
                                    type_part = parts[0]
                                    for action_type in type_part.split("+"):
                                        action_types.add(action_type.strip())
                                    
                                    # 处理行为内容部分 (冒号后)
                                    if len(parts) >= 2:
                                        content_part = parts[1]
                                        for action_content in content_part.split("+"):
                                            action_contents.add(action_content.strip())
                                
                            # 提取风格
                            style_match = re.search(style_pattern, tag)
                            if style_match:
                                label = style_match.group(1).strip()
                                keigo_type.add(label)
                                logging.debug(f"找到风格标签: {label}")
                
                processed_files += 1
                if processed_files % 10 == 0:
                    logging.info(f"已处理 {processed_files}/{file_count} 个文件")
                    
            except json.JSONDecodeError as e:
                logging.error(f"解析JSON文件 {file_path} 时出错: {str(e)}")
            except Exception as e:
                logging.error(f"处理文件 {file_path} 时出错: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
    
    logging.info(f"文件处理完成。总共扫描了 {file_count} 个文件，成功处理了 {processed_files} 个文件")
    
    # 构建映射字典
    interaction_object_map = {obj: i for i, obj in enumerate(sorted(object_of_exchange))}
    interaction_role_map = {role: i for i, role in enumerate(sorted(role_in_conversation))}
    sender_action_map = {action: i for i, action in enumerate(sorted(sender_actions))}
    style_map = {style: i for i, style in enumerate(sorted(keigo_type))}
    
    # 构建拆分后的映射
    action_type_map = {act_type: i for i, act_type in enumerate(sorted(action_types))}
    action_content_map = {act_content: i for i, act_content in enumerate(sorted(action_contents))}
    
    logging.info(f"提取的标签统计:")
    logging.info(f"交互对象标签: {len(object_of_exchange)} 个")
    logging.info(f"交互角色标签: {len(role_in_conversation)} 个")
    logging.info(f"发送者行为标签: {len(sender_actions)} 个")
    logging.info(f"行为类型标签: {len(action_types)} 个")
    logging.info(f"行为内容标签: {len(action_contents)} 个")
    logging.info(f"风格标签: {len(keigo_type)} 个")
    
    # 保存映射到JSON文件
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

    logging.info(f"自动提取的标签映射已保存到 {os.path.abspath(maps_file)}")
    
    return interaction_object_map, interaction_role_map, sender_action_map, action_type_map, action_content_map, style_map

class Config:
    def __init__(self):
        # 先检查命令行，看看是不是推理模式
        mode = None
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]

        # 如果是inference模式，就不创建新目录，直接给个空字符串或None都行
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
        
        # 从自动生成的映射文件加载Stage2和Stage3的标签映射
        self._load_auto_label_maps()
        
    def _load_auto_label_maps(self):
        """从自动生成的映射文件加载标签映射"""
        maps_file = "./label_maps.json"

        if not os.path.exists(maps_file):
            logging.error(f"自动标签映射文件 {maps_file} 不存在")
            raise FileNotFoundError(f"找不到标签映射文件，请先运行 'python classifier.py --mode build_maps --data_dir ./data' 生成映射文件")
        
        try:
            logging.info(f"从 {maps_file} 加载自动标签映射")
            with open(maps_file, "r", encoding="utf-8") as f:
                maps = json.load(f)
                
            # 加载所有映射
            self.interaction_object_map = maps["interaction_object_map"]
            self.interaction_role_map = maps["interaction_role_map"]
            self.sender_action_map = maps["sender_action_map"]
            self.action_type_map = maps["action_type_map"]
            self.action_content_map = maps["action_content_map"]
            self.style_map = maps["style_map"]
            
            # 记录加载的标签数量
            logging.info(f"已加载标签映射:")
            logging.info(f"交互对象标签: {len(self.interaction_object_map)} 个")
            logging.info(f"交互角色标签: {len(self.interaction_role_map)} 个")
            logging.info(f"发送者行为标签: {len(self.sender_action_map)} 个")
            logging.info(f"行为类型标签: {len(self.action_type_map)} 个")
            logging.info(f"行为内容标签: {len(self.action_content_map)} 个")
            logging.info(f"风格标签: {len(self.style_map)} 个")
            
        except Exception as e:
            logging.error(f"加载自动标签映射时出错: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError("无法加载自动标签映射文件，请先运行 build_maps 模式生成映射文件")

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
    自动识别标签类型并更新相应的向量。
    """
    # 初始化结果向量
    vec = {
        "interaction_object": [0] * len(interaction_object_map),
        "interaction_role": [0] * len(interaction_role_map),
        "sender_action": [0] * len(sender_action_map)
    }
    
    # 定义标签类型的正则表达式模式
    object_pattern = r"第二層:やり取りされるもの:(.*)"
    role_pattern = r"第二層:やり取りにおける役割:(.*)"
    action_pattern = r"第二層:送信者の動き:(.*)"
    
    for tag in tags:
        if not isinstance(tag, str):
            continue
            
        # 匹配交互对象
        object_match = re.search(object_pattern, tag)
        if object_match:
            label_str = object_match.group(1).strip()
            if label_str in interaction_object_map:
                idx = interaction_object_map[label_str]
                vec["interaction_object"][idx] = 1
            else:
                # 如果标签不在映射中，记录日志
                logging.warning(f"未知的交互对象标签: {label_str}")
                
        # 匹配交互角色
        role_match = re.search(role_pattern, tag)
        if role_match:
            label_str = role_match.group(1).strip()
            if label_str in interaction_role_map:
                idx = interaction_role_map[label_str]
                vec["interaction_role"][idx] = 1
            else:
                logging.warning(f"未知的交互角色标签: {label_str}")
                
        # 匹配发送者行为
        action_match = re.search(action_pattern, tag)
        if action_match:
            label_str = action_match.group(1).strip()
            if label_str in sender_action_map:
                idx = sender_action_map[label_str]
                vec["sender_action"][idx] = 1
            else:
                logging.warning(f"未知的发送者行为标签: {label_str}")
    
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
    logging.info(f"Stage2加载样本数量: {len(samples)}")
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
    logging.info(f"Stage3加载样本数量: {len(samples)}")
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
    # 不再指定具体的日志文件名，让setup_logger自动生成
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
            
            # 訓練精度計算
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
    多ラベル分類の逐要素精度を計算します
    """
    preds_bin = (preds >= threshold).float()
    correct = (preds_bin == labels).float().sum().item()
    total = labels.numel()
    return correct / total

def compute_multilabel_accuracy_detailed(preds, labels, label_map, threshold=0.5):
    """
    计算多标签分类的详细精度，返回每个标签的精度
    """
    preds_bin = (preds >= threshold).float()
    per_label_correct = {}
    per_label_total = {}
    per_label_acc = {}
    
    # 初始化计数器
    for label in label_map:
        per_label_correct[label] = 0
        per_label_total[label] = 0
    
    # 计算每个标签的正确数和总数
    for i, label_name in enumerate(label_map):
        idx = label_map[label_name]
        correct = (preds_bin[:, idx] == labels[:, idx]).float().sum().item()
        total = labels.size(0)
        per_label_correct[label_name] = correct
        per_label_total[label_name] = total
        per_label_acc[label_name] = correct / total * 100 if total > 0 else 0
    
    # 计算总体精度
    correct = (preds_bin == labels).float().sum().item()
    total = labels.numel()
    overall_acc = correct / total * 100 if total > 0 else 0
    
    return overall_acc, per_label_acc

def train_stage2(config):
    setup_logger()
    # 首先打印所有标签映射信息
    logging.info("\n===== 标签映射详细信息 =====")
    
    # 打印交互对象映射
    logging.info("\n交互对象映射 (interaction_object_map):")
    for label, idx in sorted(config.interaction_object_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # 打印交互角色映射
    logging.info("\n交互角色映射 (interaction_role_map):")
    for label, idx in sorted(config.interaction_role_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # 打印发送者行为映射
    logging.info("\n发送者行为映射 (sender_action_map):")
    for label, idx in sorted(config.sender_action_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    # 如果存在行为类型和内容映射，也打印它们
    if hasattr(config, 'action_type_map'):
        logging.info("\n行为类型映射 (action_type_map):")
        for label, idx in sorted(config.action_type_map.items(), key=lambda x: x[1]):
            logging.info(f"  {idx}: {label}")
    
    if hasattr(config, 'action_content_map'):
        logging.info("\n行为内容映射 (action_content_map):")
        for label, idx in sorted(config.action_content_map.items(), key=lambda x: x[1]):
            logging.info(f"  {idx}: {label}")
    
    logging.info("\n===== 标签映射打印完成 =====")
    
    logging.info("\n=== Stage2モデル（文節級多ラベル予測）の訓練開始 ===")
    
    # 确保这里不要再调用setup_logger()，因为前面已经调用过了
    # 原有训练代码从这里开始...
    
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
            
            # 訓練時の精度計算
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
            # 收集所有的预测和标签
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
                
                # 精度計算
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
            
            # 合并所有批次的预测和标签
            all_inter_obj_preds = torch.cat(all_inter_obj_preds, dim=0)
            all_inter_role_preds = torch.cat(all_inter_role_preds, dim=0)
            all_sender_act_preds = torch.cat(all_sender_act_preds, dim=0)
            all_inter_obj_labels = torch.cat(all_inter_obj_labels, dim=0)
            all_inter_role_labels = torch.cat(all_inter_role_labels, dim=0)
            all_sender_act_labels = torch.cat(all_sender_act_labels, dim=0)
            
            # 计算详细的精度
            obj_acc, obj_per_label_acc = compute_multilabel_accuracy_detailed(
                all_inter_obj_preds, all_inter_obj_labels, config.interaction_object_map)
            role_acc, role_per_label_acc = compute_multilabel_accuracy_detailed(
                all_inter_role_preds, all_inter_role_labels, config.interaction_role_map)
            act_acc, act_per_label_acc = compute_multilabel_accuracy_detailed(
                all_sender_act_preds, all_sender_act_labels, config.sender_action_map)
            
            # 输出详细的精度报告
            logging.info(f"\n===== 详细标签精度报告 =====")
            
            logging.info("\n交互对象标签精度:")
            for label, acc in sorted(obj_per_label_acc.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {label}: {acc:.2f}%")
            
            logging.info("\n交互角色标签精度:")
            for label, acc in sorted(role_per_label_acc.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"  {label}: {acc:.2f}%")
            
            logging.info("\n发送者行为标签精度:")
            # 仅显示前20个和后5个，以避免过多输出
            sorted_act_accs = sorted(act_per_label_acc.items(), key=lambda x: x[1], reverse=True)
            logging.info(f"  前20个最高精度的标签:")
            for label, acc in sorted_act_accs[:20]:
                logging.info(f"  {label}: {acc:.2f}%")
            
            if len(sorted_act_accs) > 20:
                logging.info(f"\n  后5个最低精度的标签:")
                for label, acc in sorted_act_accs[-5:]:
                    logging.info(f"  {label}: {acc:.2f}%")
            
            # 随机选择几个样本并显示它们的预测和实际标签
            logging.info("\n===== 样本预测示例 =====")
            try:
                sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
                
                for idx in sample_indices:
                    try:
                        sample = val_dataset[idx]
                        logging.info(f"\n样本 #{idx}:")
                        logging.info(f"样本键: {list(sample.keys())}")
                        
                        # 解码邮件内容
                        mail_ids = sample["mail_input_ids"].tolist()
                        mail_ids_clean = [tid for tid in mail_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                        if mail_ids_clean:
                            mail_text = tokenizer.decode(mail_ids_clean)
                            logging.info(f"邮件内容: {mail_text[:100]}..." if len(mail_text) > 100 else mail_text)
                        else:
                            logging.info("无法解码邮件内容")
                        
                        # 解码句子内容
                        sent_ids = sample["sent_input_ids"].tolist()
                        sent_ids_clean = [tid for tid in sent_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                        if sent_ids_clean:
                            sent_text = tokenizer.decode(sent_ids_clean)
                            logging.info(f"句子内容: {sent_text}")
                        else:
                            logging.info("无法解码句子内容")
                        
                        # 使用已有的tokenized输入，直接进行模型预测
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
                        
                        # 获取预测概率
                        obj_probs = torch.sigmoid(inter_obj_logits)[0].cpu().numpy()
                        role_probs = torch.sigmoid(inter_role_logits)[0].cpu().numpy()
                        act_probs = torch.sigmoid(sender_act_logits)[0].cpu().numpy()
                        
                        # 显示真实标签
                        logging.info("真实标签:")
                        
                        logging.info("  交互对象:")
                        true_obj_indices = np.where(sample["interaction_object"].numpy() == 1)[0]
                        for i in true_obj_indices:
                            label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        logging.info("  交互角色:")
                        true_role_indices = np.where(sample["interaction_role"].numpy() == 1)[0]
                        for i in true_role_indices:
                            label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        logging.info("  发送者行为:")
                        true_act_indices = np.where(sample["sender_action"].numpy() == 1)[0]
                        for i in true_act_indices:
                            label = [k for k, v in config.sender_action_map.items() if v == i][0]
                            logging.info(f"    - {label}")
                        
                        # 显示预测结果
                        logging.info("\n预测结果:")
                        
                        # 显示交互对象预测
                        logging.info("  交互对象 (前3):")
                        top_obj_indices = np.argsort(obj_probs)[::-1][:3]
                        for i in top_obj_indices:
                            label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                            is_true = i in true_obj_indices
                            logging.info(f"    - {label}: {obj_probs[i]:.4f} {'✓' if is_true else '✗'}")
                        
                        # 显示交互角色预测
                        logging.info("  交互角色 (前3):")
                        top_role_indices = np.argsort(role_probs)[::-1][:3]
                        for i in top_role_indices:
                            label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                            is_true = i in true_role_indices
                            logging.info(f"    - {label}: {role_probs[i]:.4f} {'✓' if is_true else '✗'}")
                        
                        # 显示发送者行为预测
                        logging.info("  发送者行为 (前5):")
                        top_act_indices = np.argsort(act_probs)[::-1][:5]
                        for i in top_act_indices:
                            label = [k for k, v in config.sender_action_map.items() if v == i][0]
                            is_true = i in true_act_indices
                            logging.info(f"    - {label}: {act_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    except Exception as e:
                        logging.error(f"处理样本 #{idx} 时出错: {str(e)}")
                        logging.error(traceback.format_exc())
            except Exception as e:
                logging.error(f"样本预测部分出错: {str(e)}")
                logging.error(traceback.format_exc())
            
            logging.info("===== 详细报告结束 =====")
        
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
        
        logging.info("\n=== Stage2训练完成 ===")
        logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
        
        # 直接在验证完成后添加详细报告，不要再嵌套一个epoch循环
        logging.info("\n===== 标签详细精度报告 =====")
        
        # 收集所有验证数据的预测和标签
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
                
                # Stage1预测
                keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
                stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
                stage1_role_label = torch.softmax(role_logits, dim=1)
                
                # Stage2预测
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
            
            # 合并所有批次的预测和标签
            all_obj_preds = torch.cat(all_obj_preds, dim=0)
            all_role_preds = torch.cat(all_role_preds, dim=0)
            all_act_preds = torch.cat(all_act_preds, dim=0)
            all_obj_labels = torch.cat(all_obj_labels, dim=0)
            all_role_labels = torch.cat(all_role_labels, dim=0)
            all_act_labels = torch.cat(all_act_labels, dim=0)
        
        # 计算每个小标签的精度
        # 交互对象标签精度
        logging.info("\n交互对象标签精度详情:")
        for label, idx in config.interaction_object_map.items():
            true_pos = ((all_obj_preds[:, idx] >= 0.5) & (all_obj_labels[:, idx] == 1)).sum().item()
            true_neg = ((all_obj_preds[:, idx] < 0.5) & (all_obj_labels[:, idx] == 0)).sum().item()
            false_pos = ((all_obj_preds[:, idx] >= 0.5) & (all_obj_labels[:, idx] == 0)).sum().item()
            false_neg = ((all_obj_preds[:, idx] < 0.5) & (all_obj_labels[:, idx] == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = (all_obj_labels[:, idx] == 1).sum().item()
            
            logging.info(f"  {label}: 精度={precision*100:.2f}%, 召回率={recall*100:.2f}%, F1={f1*100:.2f}%, 支持数={support}")
        
        # 交互角色标签精度
        logging.info("\n交互角色标签精度详情:")
        for label, idx in config.interaction_role_map.items():
            true_pos = ((all_role_preds[:, idx] >= 0.5) & (all_role_labels[:, idx] == 1)).sum().item()
            true_neg = ((all_role_preds[:, idx] < 0.5) & (all_role_labels[:, idx] == 0)).sum().item()
            false_pos = ((all_role_preds[:, idx] >= 0.5) & (all_role_labels[:, idx] == 0)).sum().item()
            false_neg = ((all_role_preds[:, idx] < 0.5) & (all_role_labels[:, idx] == 1)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = (all_role_labels[:, idx] == 1).sum().item()
            
            logging.info(f"  {label}: 精度={precision*100:.2f}%, 召回率={recall*100:.2f}%, F1={f1*100:.2f}%, 支持数={support}")
        
        # 发送者行为标签精度（可能很多，只显示前20个支持数最多的）
        logging.info("\n发送者行为标签精度详情 (按支持数排序前20):")
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
            logging.info(f"  {label}: 精度={precision*100:.2f}%, 召回率={recall*100:.2f}%, F1={f1*100:.2f}%, 支持数={support}")
        
        # 显示几个样本的具体预测结果
        logging.info("\n===== 样本预测示例 =====")
        try:
            sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
            
            for idx in sample_indices:
                try:
                    sample = val_dataset[idx]
                    logging.info(f"\n样本 #{idx}:")
                    logging.info(f"样本键: {list(sample.keys())}")
                    
                    # 解码邮件内容
                    mail_ids = sample["mail_input_ids"].tolist()
                    mail_ids_clean = [tid for tid in mail_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                    if mail_ids_clean:
                        mail_text = tokenizer.decode(mail_ids_clean)
                        logging.info(f"邮件内容: {mail_text[:100]}..." if len(mail_text) > 100 else mail_text)
                    else:
                        logging.info("无法解码邮件内容")
                    
                    # 解码句子内容
                    sent_ids = sample["sent_input_ids"].tolist()
                    sent_ids_clean = [tid for tid in sent_ids if tid not in [0, tokenizer.cls_token_id, tokenizer.sep_token_id]]
                    if sent_ids_clean:
                        sent_text = tokenizer.decode(sent_ids_clean)
                        logging.info(f"句子内容: {sent_text}")
                    else:
                        logging.info("无法解码句子内容")
                    
                    # 使用已有的tokenized输入，直接进行模型预测
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
                    
                    # 显示真实标签
                    logging.info("真实标签:")
                    
                    logging.info("  交互对象:")
                    true_obj_indices = np.where(sample["interaction_object"].numpy() == 1)[0]
                    for i in true_obj_indices:
                        label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    logging.info("  交互角色:")
                    true_role_indices = np.where(sample["interaction_role"].numpy() == 1)[0]
                    for i in true_role_indices:
                        label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    logging.info("  发送者行为:")
                    true_act_indices = np.where(sample["sender_action"].numpy() == 1)[0]
                    for i in true_act_indices:
                        label = [k for k, v in config.sender_action_map.items() if v == i][0]
                        logging.info(f"    - {label}")
                    
                    # 显示预测结果
                    logging.info("\n预测结果:")
                    
                    logging.info("  交互对象 (前3):")
                    top_obj_indices = np.argsort(obj_probs)[::-1][:3]
                    for i in top_obj_indices:
                        label = [k for k, v in config.interaction_object_map.items() if v == i][0]
                        is_true = i in true_obj_indices
                        logging.info(f"    - {label}: {obj_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    logging.info("  交互角色 (前3):")
                    top_role_indices = np.argsort(role_probs)[::-1][:3]
                    for i in top_role_indices:
                        label = [k for k, v in config.interaction_role_map.items() if v == i][0]
                        is_true = i in true_role_indices
                        logging.info(f"    - {label}: {role_probs[i]:.4f} {'✓' if is_true else '✗'}")
                    
                    logging.info("  发送者行为 (前5):")
                    top_act_indices = np.argsort(act_probs)[::-1][:5]
                    for i in top_act_indices:
                        label = [k for k, v in config.sender_action_map.items() if v == i][0]
                        is_true = i in true_act_indices
                        logging.info(f"    - {label}: {act_probs[i]:.4f} {'✓' if is_true else '✗'}")
                
                except Exception as e:
                    logging.error(f"处理样本 #{idx} 时出错: {str(e)}")
                    logging.error(traceback.format_exc())
        except Exception as e:
            logging.error(f"样本预测部分出错: {str(e)}")
            logging.error(traceback.format_exc())
        
        logging.info("\n===== 详细报告结束 =====")

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
        # 收集所有的测试集预测和标签
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
            
            # 精度計算
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
        
        # 合并所有批次的预测和标签
        all_inter_obj_preds = torch.cat(all_inter_obj_preds, dim=0)
        all_inter_role_preds = torch.cat(all_inter_role_preds, dim=0)
        all_sender_act_preds = torch.cat(all_sender_act_preds, dim=0)
        all_inter_obj_labels = torch.cat(all_inter_obj_labels, dim=0)
        all_inter_role_labels = torch.cat(all_inter_role_labels, dim=0)
        all_sender_act_labels = torch.cat(all_sender_act_labels, dim=0)
        
        # 计算详细的精度
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
    
    # 输出详细的精度报告
    logging.info(f"\n===== テストデータの詳細標签精度報告 =====")
    
    logging.info("\n交互对象标签精度:")
    for label, acc in sorted(obj_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n交互角色标签精度:")
    for label, acc in sorted(role_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n发送者行为标签精度:")
    # 仅显示前20个和后5个，以避免过多输出
    sorted_act_accs = sorted(act_per_label_acc.items(), key=lambda x: x[1], reverse=True)
    logging.info(f"  前20个最高精度的标签:")
    for label, acc in sorted_act_accs[:20]:
        logging.info(f"  {label}: {acc:.2f}%")
    
    if len(sorted_act_accs) > 20:
        logging.info(f"\n  后5个最低精度的标签:")
        for label, acc in sorted_act_accs[-5:]:
            logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n=== Stage2训练完成 ===")
    logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
    logging.info(f"テスト総合精度: {avg_test_acc:.2f}%")

def train_stage3(config):
    # 先打印所有标签映射
    logging.info("===== Stage3训练开始 =====")
    logging.info("自动标签映射详细内容:")
    
    # 打印风格映射
    logging.info("\n风格映射 (style_map):")
    for label, idx in sorted(config.style_map.items(), key=lambda x: x[1]):
        logging.info(f"  {idx}: {label}")
    
    logging.info("\n===== 标签映射打印完成 =====")
    
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
        # 收集所有的测试集预测和标签
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
            
            # 精度計算
            style_acc = compute_multilabel_accuracy(torch.sigmoid(style_logits), batch["style_label"].to(config.device))
            test_style_acc += style_acc
            test_batches += 1
            
            all_style_preds.append(torch.sigmoid(style_logits).cpu())
            all_style_labels.append(batch["style_label"].cpu())
        
        # 合并所有批次的预测和标签
        all_style_preds = torch.cat(all_style_preds, dim=0)
        all_style_labels = torch.cat(all_style_labels, dim=0)
        
        # 计算详细的精度
        style_acc, style_per_label_acc = compute_multilabel_accuracy_detailed(
            all_style_preds, all_style_labels, config.style_map)
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_style_acc = test_style_acc / test_batches * 100
    
    logging.info(f"\nStage3 テスト結果:")
    logging.info(f"  Test Loss = {avg_test_loss:.4f}")
    logging.info(f"  スタイル精度: {avg_test_style_acc:.2f}%")
    
    # 输出详细的精度报告
    logging.info(f"\n===== テストデータの詳細標签精度報告 =====")
    
    logging.info("\n风格标签精度:")
    for label, acc in sorted(style_per_label_acc.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  {label}: {acc:.2f}%")
    
    logging.info("\n=== Stage3训练完成 ===")
    logging.info(f"最終最高検証精度: {best_val_acc:.2f}%")
    logging.info(f"テスト総合精度: {avg_test_style_acc:.2f}%")

#####################################
# 7. 推論（Inference）関数            #
#####################################

def inference_pipeline(config, input_json_path, model_dir):
    """
    推理管道，将输入JSON处理并生成预测结果
    
    参数:
      config: 配置对象
      input_json_path: 输入JSON文件路径
      model_dir: 模型目录，包含所有三个阶段的模型
    """
    setup_logger()
    print(f"\n===== 分類開始 =====")
    print(f"输入文件: {input_json_path}")
    print(f"模型目录: {model_dir}")
    
    # 所有模型都存在同一目录下，使用不同的文件名
    stage1_model_path = os.path.join(model_dir, "stage1_model.bin")
    stage2_model_path = os.path.join(model_dir, "stage2_model.bin")
    stage3_model_path = os.path.join(model_dir, "stage3_model.bin")
    
    # 检查模型文件是否存在
    for model_path in [stage1_model_path, stage2_model_path, stage3_model_path]:
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return
    
    try:
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = BertJapaneseTokenizer.from_pretrained(config.pretrained_model)
        
        # 加载Stage1模型
        print("加载Stage1模型...")
        stage1_model = Stage1Model(config.pretrained_model, 
                                  num_keigo=len(config.social_standing_results_map), 
                                  num_role=len(config.role_pair_map))
        stage1_model.load_state_dict(torch.load(stage1_model_path, map_location=config.device))
        stage1_model.to(config.device)
        stage1_model.eval()
        
        # 加载Stage2模型
        print("加载Stage2模型...")
        stage2_model = Stage2Model(config.pretrained_model,
                                  num_inter_obj=len(config.interaction_object_map),
                                  num_inter_role=len(config.interaction_role_map),
                                  num_sender_act=len(config.sender_action_map),
                                  num_social_standing_resultss=len(config.social_standing_results_map),
                                  num_role_labels=len(config.role_pair_map))
        stage2_model.load_state_dict(torch.load(stage2_model_path, map_location=config.device))
        stage2_model.to(config.device)
        stage2_model.eval()
        
        # 加载Stage3模型
        print("加载Stage3模型...")
        stage3_model = Stage3Model(config.pretrained_model,
                                  num_keigo_type=len(config.style_map),
                                  num_inter_obj=len(config.interaction_object_map),
                                  num_inter_role=len(config.interaction_role_map),
                                  num_sender_act=len(config.sender_action_map))
        stage3_model.load_state_dict(torch.load(stage3_model_path, map_location=config.device))
        stage3_model.to(config.device)
        stage3_model.eval()
        
        # 读取输入JSON
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取邮件文本
        mail_text = ""
        if "mail_text" in data:
            mail_text = data["mail_text"]
        elif "本文" in data:
            # 尝试从本文字段提取文本
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
        
        # 如果没有找到邮件文本，尝试使用assemble_mail_text函数
        if not mail_text and "assemble_mail_text" in globals():
            mail_text = assemble_mail_text(data)
        
        if not mail_text:
            print("错误: 无法从输入JSON中提取邮件文本")
            return
        
        # 提取句子
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
            # 如果没有找到句子，将整个邮件作为一个句子
            sentences = [mail_text]
        
        # 打印邮件内容
        print(f"\n===== メール内容 =====")
        print(mail_text)
        print(f"\n{len(sentences)} 個の文を分析する必要があります")
        
        # 对整个邮件进行一次Stage1预测
        mail_enc = tokenizer(mail_text, truncation=True, padding='max_length', 
                            max_length=config.max_length, return_tensors="pt")
        mail_input_ids = mail_enc["input_ids"].to(config.device)
        mail_attention_mask = mail_enc["attention_mask"].to(config.device)
        
        with torch.no_grad():
            # Stage1预测（对整个邮件）
            keigo_logits, role_logits = stage1_model(mail_input_ids, mail_attention_mask)
            keigo_probs = torch.softmax(keigo_logits, dim=1)[0].cpu().numpy()
            role_probs = torch.softmax(role_logits, dim=1)[0].cpu().numpy()
            
            # 获取预测标签
            keigo_pred = np.argmax(keigo_probs)
            role_pred = np.argmax(role_probs)
            
            # 获取标签名称
            social_standing_results = [k for k, v in config.social_standing_results_map.items() if v == keigo_pred][0]
            role_label = [k for k, v in config.role_pair_map.items() if v == role_pred][0]
            
            # 打印Stage1预测结果
            print(f"\n===== メール全体レベル予測結果 =====")
            print(f"社会関係: {social_standing_results} (確率: {keigo_probs[keigo_pred]:.4f})")
            print(f"役割関係: {role_label} (確率: {role_probs[role_pred]:.4f})")
            
            # 将Stage1预测结果转换为张量，用于后续阶段
            stage1_social_standing_results = torch.softmax(keigo_logits, dim=1)
            stage1_role_label = torch.softmax(role_logits, dim=1)
        
        # 处理每个句子
        results = []
        for i, sentence in enumerate(sentences):
            print(f"\n===== 処理する文 {i+1}/{len(sentences)} =====")
            print(f"文の内容: {sentence}")
            
            # 对句子进行编码
            sent_enc = tokenizer(sentence, truncation=True, padding='max_length', 
                                max_length=config.max_length, return_tensors="pt")
            sent_input_ids = sent_enc["input_ids"].to(config.device)
            sent_attention_mask = sent_enc["attention_mask"].to(config.device)
            
            with torch.no_grad():
                # Stage2预测（使用整个邮件的Stage1预测结果）
                inter_obj_logits, inter_role_logits, sender_act_logits = stage2_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage1_social_standing_results, stage1_role_label
                )
                
                obj_probs = torch.sigmoid(inter_obj_logits)[0].cpu().numpy()
                role_probs = torch.sigmoid(inter_role_logits)[0].cpu().numpy()
                act_probs = torch.sigmoid(sender_act_logits)[0].cpu().numpy()
                
                # Stage3预测
                stage2_inter_obj = torch.sigmoid(inter_obj_logits)
                stage2_inter_role = torch.sigmoid(inter_role_logits)
                stage2_sender_act = torch.sigmoid(sender_act_logits)
                
                style_logits = stage3_model(
                    mail_input_ids, mail_attention_mask,
                    sent_input_ids, sent_attention_mask,
                    stage2_inter_obj, stage2_inter_role, stage2_sender_act
                )
                
                style_probs = torch.sigmoid(style_logits)[0].cpu().numpy()
            
            # 获取交互对象的前3个预测
            top_objects = []
            top_obj_indices = np.argsort(obj_probs)[::-1][:3]
            for idx in top_obj_indices:
                label = [k for k, v in config.interaction_object_map.items() if v == idx][0]
                top_objects.append({"label": label, "probability": float(obj_probs[idx])})
            
            # 获取交互角色的前3个预测
            top_roles = []
            top_role_indices = np.argsort(role_probs)[::-1][:3]
            for idx in top_role_indices:
                label = [k for k, v in config.interaction_role_map.items() if v == idx][0]
                top_roles.append({"label": label, "probability": float(role_probs[idx])})
            
            # 获取发送者行为的前5个预测
            top_actions = []
            top_act_indices = np.argsort(act_probs)[::-1][:5]
            for idx in top_act_indices:
                label = [k for k, v in config.sender_action_map.items() if v == idx][0]
                top_actions.append({"label": label, "probability": float(act_probs[idx])})
            
            # 获取风格标签
            keigo_type_results = []
            for idx, (style_name, _) in enumerate(config.style_map.items()):
                keigo_type_results.append({"style": style_name, "probability": float(style_probs[idx])})
            
            # 输出结果
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
            
            # 将结果添加到列表
            sentence_result = {
                "sentence": sentence,
                "social_standing": social_standing_results,  # 使用邮件级别的预测结果
                "role_relationship": role_label,  # 使用邮件级别的预测结果
                "object_of_exchange": top_objects,
                "role_in_conversation": top_roles,
                "sender_actions": top_actions,
                "keigo_type": keigo_type_results
            }
            results.append(sentence_result)
        
        # 创建完整的结果对象
        final_result = {
            "mail_text": mail_text,
            "mail_level_prediction": {
                "keigo_type": social_standing_results,
                "role_relation": role_label
            },
            "sentence_results": results
        }
        
        # 保存结果到JSON
        output_path = input_json_path.replace(".json", "_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n分類完了，結果は {output_path} に保存されました")
        
        # 打印总结
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
    parser.add_argument("--auto_maps", action="store_true", help="自動的に標签映射を構築する")
    
    args = parser.parse_args()
    
    set_seed(42)
    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir

    if args.mode == "build_maps":
        setup_logger()
        logging.info("开始构建标签映射...")
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
            print("エラー: 推論モードでは --input パラメータが必要です")
            return
        if not args.model_dir:
            print("エラー: 推論モードでは --model_dir パラメータが必要です")
            return
        # 这里使用正确的函数名inference_pipeline
        inference_pipeline(config, args.input, args.model_dir)

if __name__ == "__main__":
    main()
