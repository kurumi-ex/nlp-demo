import os
from pathlib import Path
import re
from typing import Literal
import requests
from enum import Enum
from collections import Counter

from ch8.timemachine.timemachine_process import Vocabulary, ST


def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def get_raw_nmt():
    file_path = './fra-eng/fra.txt'

    with open(file_path, 'r', encoding='utf-8') as f:
        return preprocess_nmt(f.read())


def tokenize_nmt(text, seq_len=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if seq_len and i >= seq_len:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def flatten(src_list):
    res = []
    for item in src_list:
        res.extend(item)
    return res


raw_nmt = get_raw_nmt()
s, t = tokenize_nmt(raw_nmt)
src_vocab = Vocabulary(flatten(s), [ST.UNK, ST.PAD, ST.SOS, ST.EOS])
print(src_vocab.vocabulary)
