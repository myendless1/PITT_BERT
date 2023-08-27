import torch
from transformers import BertConfig, BertTokenizer
from transformers import BertModel

if __name__ == '__main__':
    model_config = BertConfig.from_pretrained('models/BERT/bert-tiny/bert_config.json')
    model_path = 'models/BERT/bert-tiny'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(model_path, config=model_config, ignore_mismatched_sizes=True)

    hidden_dimension = 64
    grid_len = 100

    seq1 = "i would like to use bert"
    seq2 = "you are not willing to use bert"
    seqs = []
    seqs.append(seq1)
    seqs.append(seq2)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    seq_out = tokenizer(seqs)
    input_ids = seq_out['input_ids']
    input_ids = torch.tensor([lst + [0] * (500 - len(lst)) for lst in input_ids])

    token_type_ids = seq_out['token_type_ids']
    token_type_ids = torch.tensor([lst + [0] * (500 - len(lst)) for lst in token_type_ids])

    attention_mask = seq_out['attention_mask']
    attention_mask = torch.tensor([lst + [0] * (500 - len(lst)) for lst in attention_mask])

    embedding_linear = torch.nn.Linear(in_features=500, out_features=grid_len)
    hidden_linear = torch.nn.Linear(in_features=128, out_features=hidden_dimension)

    bert_out = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    out = hidden_linear(bert_out['hidden_states'][0])

    print(out)
