"""
utils/imagenet_labels.py
提供 ImageNet-1K 人类可读标签列表，供 Streamlit 下拉选择框使用。
"""

import json
import os
import urllib.request
from typing import List

LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)
# 本地标签文件路径（HF Spaces 部署时优先使用，避免网络依赖）
LOCAL_LABELS_PATH = os.path.join(os.path.dirname(__file__), "imagenet_labels.json")


def load_labels() -> List[str]:
    """优先从本地加载 ImageNet 标签，失败时回退为网络下载或数字 ID 列表。"""
    try:
        local_path = os.path.abspath(LOCAL_LABELS_PATH)
        if os.path.isfile(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    try:
        with urllib.request.urlopen(LABELS_URL, timeout=10) as response:
            labels = json.loads(response.read().decode("utf-8"))
        return labels
    except Exception:
        return [str(i) for i in range(1000)]


def get_label_options() -> List[str]:
    """返回带索引前缀的标签列表，如 ['0: tench', '1: goldfish', ...]。"""
    labels = load_labels()
    return [f"{i}: {name}" for i, name in enumerate(labels)]


def parse_label_option(option: str) -> int:
    """从 '7: cock' 格式的字符串中解析出类别索引。"""
    return int(option.split(":")[0])
