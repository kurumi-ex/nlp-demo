import torch

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


def get_seq(vocab: Vocabulary, slist):
    res = []
    valid_len = []
    for item in slist:
        valid_len.append(len(item))
        res.append([vocab.get_index(e) for e in item] + [vocab.get_index(str(ST.EOS))])
    return res, valid_len


def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    else:
        return line + [padding_token] * (num_steps - len(line))


def get_data(min_freq=1, time_steps=20, voc=None):
    raw_nmt = get_raw_nmt()
    s, t = tokenize_nmt(raw_nmt)
    # s 为 2维的列表
    src_vocab = Vocabulary(flatten(s), [ST.UNK, ST.PAD, ST.SOS, ST.EOS], min_freq=min_freq)
    tar_vocab = Vocabulary(flatten(t), [ST.UNK, ST.PAD, ST.SOS, ST.EOS], min_freq=min_freq)
    if voc:
        return src_vocab, tar_vocab

    src_raw_seq, src_len = get_seq(src_vocab, s)
    pad_token = src_vocab.get_index(str(ST.PAD))
    src_pad_seq = [truncate_pad(e, time_steps, pad_token) for e in src_raw_seq]
    src_tensor = torch.LongTensor(src_pad_seq)
    src_len = torch.LongTensor([e if e < time_steps else time_steps for e in src_len])

    tar_raw_seq, tar_len = get_seq(tar_vocab, t)
    pad_token = tar_vocab.get_index(str(ST.PAD))
    tar_pad_seq = [truncate_pad(e, time_steps, pad_token) for e in tar_raw_seq]
    tar_tensor = torch.LongTensor(tar_pad_seq)
    tar_len = torch.LongTensor([e if e < time_steps else time_steps for e in tar_len])
    return src_vocab, tar_vocab, src_tensor, tar_tensor, src_len, tar_len
