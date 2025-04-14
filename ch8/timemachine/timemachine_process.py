import os
from pathlib import Path
import re
from typing import Literal
import requests
from enum import Enum
from collections import Counter


def remove_non_alpha_and_lower(string: str) -> str:
    """去除字符串中的非字母字符、两端的空格，并转换为小写"""
    return re.sub(r'[^a-zA-Z]+', ' ', string).strip().lower()


def get_timemachine_lines() -> list[str]:
    url = r'https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    root_path = str(Path(__file__).resolve().parent)
    file_path = root_path + '/timemachine.txt'

    if not os.path.exists(file_path):
        print(f'下载：{url!r}……')
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'下载失败，错误代码：{response.status_code}')

    with open(file_path, 'r') as f:
        return [remove_non_alpha_and_lower(line) for line in f]


def tokenize(text: str, token_type: Literal['word', 'char']) -> list[str]:
    if token_type == 'word':
        return text.split()
    elif token_type == 'char':
        return list(text)
    else:
        print(f'未知的token_type={token_type!r}')


class ST(Enum):
    """特殊词元 (Special Token)"""

    UNK = '未被纳入词表的未知词元 (unknown token)'
    PAD = '用于填充序列长度的标记 (padding token)'
    SOS = '序列的开始标记 (start of sequence)'
    EOS = '序列的结束标记 (end of sequence)'
    SEP = '段落的分隔标记 (separator token)'
    MASK = '掩盖以用于预测的标记 (mask token)'

    def __str__(self):
        return f'<{self.name}>'

    @property
    def description(self):
        return self.value


# 将单词转换为编号
class Vocabulary:
    def __init__(self, flat_tokens: list[str], special_tokens, min_freq=1):
        """
        :param flat_tokens: 被展平后的词元列表
        :param min_freq: 纳入词表的最小词频
        :param special_tokens: 由特殊词元组成的可迭代对象
        """
        self._valid_token_freq = {key: cnt for key, cnt in Counter(flat_tokens).items() if cnt >= min_freq}

        tmp: list[str] = list(dict.fromkeys([str(i) for i in special_tokens]))
        tmp.extend([
            word for word, cnt in
            sorted(self._valid_token_freq.items(), key=lambda x: x[1], reverse=True)
        ])

        self._token_to_index = {}
        self._index_to_token = {}
        for i, t in enumerate(tmp):
            self._token_to_index[t] = i
            self._index_to_token[i] = t

    def __len__(self):
        """词表大小"""
        return len(self._token_to_index)

    def __repr__(self) -> str:
        return f'<Vocabulary({len(self._token_to_index)})>'

    def get_index(self, token: str) -> int:
        """根据词元获取索引值"""
        return self._token_to_index.get(token, self._token_to_index[str(ST.UNK)])

    def get_token(self, idx: int) -> str:
        """根据索引值获取词元"""
        return self._index_to_token.get(idx, str(ST.UNK))

    def encode(self, tokens: list[str]) -> list[int]:
        """将词元列表转换为索引值列表"""
        return [self.get_index(token) for token in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        """将索引值列表转换为词元列表"""
        return [self.get_token(idx) for idx in indices]

    @property
    def vocabulary(self) -> dict[str, int]:
        return self._token_to_index

    @property
    def valid_token_freq(self) -> dict[str, int]:
        """获取词元的频率字典（保持语料文本输入时的原始顺序）"""
        return self._valid_token_freq


def get_vocabulary(token_type: Literal['word', 'char'] = 'word', special_tokens=None) -> (Vocabulary, list[int]):
    """
    :param token_type: ['word', 'char'] 默认 'word'
    :param special_tokens: 枚举类ST中的
    :return:
    """
    if special_tokens is None:
        special_tokens = [ST.UNK, ST.PAD]

    # 读取行数据
    lines = get_timemachine_lines()

    # 将句子词元化
    tokens = []
    for line in lines:
        token = tokenize(line, token_type)
        tokens.extend(token)
    vocab = Vocabulary(tokens, special_tokens)
    # 语段的索引获取
    corpus = [vocab.get_index(token) for token in tokens]

    # 获得词表和语段表
    return vocab, corpus
