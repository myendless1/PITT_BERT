import json
import logging

import numpy as np
import six
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel


model_name = 'bert-base-uncased'
MODEL_PATH = 'C:\\Users\\34707\\Desktop\\PITT_BERT\\models\\BERT\\bert-base-uncased'

# a. 通过词典导入分词器
# b. 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = False
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config, ignore_mismatched_sizes=True)

print("OK")

tokenizer = BertTokenizer.from_pretrained(model_name)
# requirements: 1.uncased 2.numbers split
tokens = tokenizer.encode(
    "..1234578")

print(tokens)
decoded_tokens = tokenizer.decode(tokens)
print(tokenizer.decode(tokens))


def load_moel(model_path):
    loaded_paras = torch.load(f"{model_path}/pytorch_model.bin")
    # print(type(loaded_paras))
    # print(len(list(loaded_paras.keys())))
    # print(list(loaded_paras.keys()))
    # for name in loaded_paras.keys():
    #     print(f"### 参数名:{name},形状:{loaded_paras[name].size()}")

class BertConfig(object):
    def __init__(self,
                 vocab_size=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 pad_token_id=0,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 max_position_embedding=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_dropout_prob
        self.max_position_embedding = max_position_embedding
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r") as reader:
            text = reader.read()
        logging.info(f"成功导入BERT配置文件 {json_file}")
        return cls.from_dict(json.loads(text))


if __name__ == '__main__':
    model = load_moel(MODEL_PATH)

    hidden_dimension = 64
    grid_len = 100

    seq1 = "i would like to use bert"
    seq2 = "you are not willing to use bert"
    seqs = []
    seqs.append(seq1)
    seqs.append(seq2)

    seq_out = tokenizer(seqs)
    input_ids = seq_out['input_ids']
    input_ids = torch.tensor([lst + [0] * (500 - len(lst)) for lst in input_ids])

    token_type_ids = seq_out['token_type_ids']
    token_type_ids = torch.tensor([lst + [0] * (500 - len(lst)) for lst in token_type_ids])

    attention_mask = seq_out['attention_mask']
    attention_mask = torch.tensor([lst + [0] * (500 - len(lst)) for lst in attention_mask])

    embedding_linear = torch.nn.Linear(in_features=500, out_features=grid_len)
    hidden_linear = torch.nn.Linear(in_features=768, out_features=hidden_dimension)

    bert_out = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    out = hidden_linear(bert_out['hidden_states'][0])

    out = embedding_linear(torch.swapaxes(out['hidden_states'][0], 1, 2))

    out = torch.swapaxes(embedding_linear(torch.swapaxes(out, 1, 2)), 1, 2)

    print(out)
