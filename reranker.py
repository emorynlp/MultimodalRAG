import os
import argparse

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

def data_processing(args):
    train_data = pd.read_csv(args.train_data_path)
    val_data = pd.read_csv(args.val_data_path)
    
    train_dict = {}
    for question, context, retrieved_contexts, file_name in zip(train_data.question, train_data.context, train_data.retrieved_contexts, train_data.file_name) :
        train_dict.setdefault(file_name, [])
        retrieved_contexts = eval(retrieved_contexts)
        if context in retrieved_contexts.values() :
            train_dict[file_name].append({'question':question, 'retrieved_contexts':retrieved_contexts, 'answer':context, 'retrieve_success':True})
    
    val_dict = {}
    for question, context, retrieved_contexts, file_name in zip(val_data.question, val_data.context, val_data.retrieved_contexts, val_data.file_name) :
        val_dict.setdefault(file_name, [])
        retrieved_contexts = eval(retrieved_contexts)
        if context in retrieved_contexts.values() :
            val_dict[file_name].append({'question':question, 'retrieved_contexts':retrieved_contexts, 'answer':context, 'retrieve_success':True})
    return train_dict, val_dict

class CustomAutoModelForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(CustomAutoModelForSequenceClassification, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        
        self.attention = nn.MultiheadAttention(embed_dim=self.config.hidden_size, num_heads=8)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, token_selected=False):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_length, hidden_size)
        
        if token_selected:
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).repeat(1, hidden_states.size(1), 1)
                attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
            
            hidden_states = hidden_states.permute(1, 0, 2)
            attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
            hidden_states = attn_output.permute(1, 0, 2)
            hidden_states = self.layer_norm(hidden_states + outputs.last_hidden_state)        
        cls_representation = hidden_states[:, 0, :]
        
        x = self.dense(cls_representation)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.view(-1), labels.view(-1))

        return {"loss": loss, "logits": logits}

def average_precision_at_k(true_index, top_k_indices):
    if true_index in top_k_indices:
        rank = np.where(top_k_indices == true_index)[0][0] + 1
        return 1.0 / rank
    else:
        return 0.0

def sample_with_answer(retrieved_contexts, answer, n):
    if answer not in retrieved_contexts:
        retrieved_contexts.append(answer)    
    other_contexts = [context for context in retrieved_contexts if context != answer]
    sampled_contexts = random.sample(other_contexts, min(n - 1, len(other_contexts)))
    sampled_contexts.append(answer)
    random.shuffle(sampled_contexts)    
    return sampled_contexts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default='./sample_data/reranker/train.csv')
    parser.add_argument("--val_data_path", type=str, default='./sample_data/reranker/val.csv')
    parser.add_argument("--from_pretrained", type=str, default='BAAI/bge-m3', help='model_name or local path')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--negative_size", type=int, default=3)
    parser.add_argument("--token_k", type=int, default=3)
    parser.add_argument("--check_step", type=int, default=100)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--quit_cnt", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='./reranker_trained')
    args = parser.parse_args()

    # Data load & processing
    train_dict, val_dict = data_processing(args)

    # Define re-ranker
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    model = CustomAutoModelForSequenceClassification(model_name=args.from_pretrained, num_labels=1)
    model.cuda()

    # Hyperparameters
    epochs = args.epochs; lr = args.lr; negative_size = args.negative_size; token_k = args.token_k
    check_step = args.check_step; save_step = args.save_step; quit_cnt = args.quit_cnt

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    optimizer.zero_grad()
        
    fname_list = list(train_dict.keys())
    check_cnt = 0; stop_cnt = 0; prev_map = -np.inf
    for epoch in range(epochs) :
        if stop_cnt >= quit_cnt :
            break
        for fname in fname_list :
            if stop_cnt >= quit_cnt :
                break
            pairs = train_dict[fname]
            for pair in pairs :
                check_cnt += 1
                
                retrieve_success = pair['retrieve_success']
                question = pair['question']
                retrieved_contexts = pair['retrieved_contexts']
                retrieved_contexts = list(retrieved_contexts.values())
                answer = pair['answer']
                
                if retrieve_success :                    
                    retrieved_contexts = sample_with_answer(retrieved_contexts, answer, negative_size+1)
                    answer_index = retrieved_contexts.index(answer)        
                    for idx, context_ in enumerate(retrieved_contexts) :
                        loss = 0.0
                        torch.cuda.empty_cache()
                        
                        # score 1
                        text = f'{question} {tokenizer.sep_token} {context_}'            
                        batch = tokenizer.batch_encode_plus([text], 
                                                            max_length=tokenizer.model_max_length, 
                                                            padding='longest', 
                                                            truncation=True, 
                                                            return_tensors="pt")
                        input_ids = batch['input_ids'].cuda()
                        if answer_index == idx :
                            labels = [1]
                        else :
                            labels = [0]
                        labels = torch.Tensor([labels]).reshape(-1, 1).cuda()
                        
                        loss_ = model(input_ids=input_ids,
                                     labels=labels, 
                                     token_selected=False)['loss']
                        loss += (loss_ / len(retrieved_contexts)) * 0.7
                        input_ids.detach().cpu(); del input_ids
                        
                        # score 2
                        torch.cuda.empty_cache()
                        q_input_ids = torch.LongTensor([tokenizer.encode(question)[1:-1]]).cuda()
                        d_input_ids = torch.LongTensor([tokenizer.encode(context_)[1:-1][:tokenizer.model_max_length]]).cuda()
                        sim_matrix = torch.matmul(model.base_model(q_input_ids)['last_hidden_state'].squeeze(0), 
                                                  model.base_model(d_input_ids)['last_hidden_state'].squeeze(0).T)
                        _, top_indices = torch.topk(sim_matrix, token_k, dim=1)
                        token_selected = d_input_ids.squeeze(0)[top_indices.reshape(-1)]
                        token_selected = tokenizer.decode(token_selected)
        
                        text = f'{question} {tokenizer.sep_token} {token_selected}'
                        batch = tokenizer.batch_encode_plus([text], 
                                                            max_length=tokenizer.model_max_length, 
                                                            padding='longest', 
                                                            truncation=True, 
                                                            return_tensors="pt")
                        input_ids = batch['input_ids'].cuda()
        
                        loss_ = model(input_ids=input_ids,
                                     labels=labels, 
                                     token_selected=True)['loss']
                        loss += (loss_ / len(retrieved_contexts)) * 0.3
                        loss.backward()
                        
                        q_input_ids.detach().cpu(); del q_input_ids 
                        d_input_ids.detach().cpu(); del d_input_ids 
                        sim_matrix.detach().cpu(); del sim_matrix 
                        input_ids.detach().cpu(); del input_ids 
                    else :
                        optimizer.step()
                        optimizer.zero_grad()
           
                if (check_cnt) % check_step == 0:
                    print(check_cnt, loss.item())
    
                if check_cnt % save_step == 0 :
                    print('Evaluation..', check_cnt)
                    model.eval()
                    ap_k_lst = []
                    for fname in val_dict :
                        pairs = val_dict[fname]
                        for pair in pairs :
                            retrieve_success = pair['retrieve_success']
                            if retrieve_success :
                                question = pair['question']
                                retrieved_contexts = pair['retrieved_contexts']
                                retrieved_contexts = list(retrieved_contexts.values())
                                answer = pair['answer']
                                answer_index = retrieved_contexts.index(answer)
                        
                                scores = []
                                for context_ in retrieved_contexts :
                                    text = f'{question} {tokenizer.sep_token} {context_}'            
                                    batch = tokenizer.batch_encode_plus([text], 
                                                                        max_length=tokenizer.model_max_length, 
                                                                        padding='longest', 
                                                                        truncation=True, 
                                                                        return_tensors="pt")
                                    input_ids = batch['input_ids'].cuda()
                                    with torch.no_grad() :
                                        output = model(input_ids)['logits']
                                        score = float(output.detach().cpu().numpy()[0][0])
                                        scores.append(score)
                                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                                sorted_indices = np.array(sorted_indices)
                                ap_k = average_precision_at_k(answer_index, sorted_indices)
                                ap_k_lst.append(ap_k)
                    map_k = np.mean(ap_k_lst)
                    print('Step', check_cnt, '\t', map_k)
                    if map_k > prev_map :
                        torch.save(model.state_dict(), f'{save_dir}/best.pt')
                        print(f'[saved] {save_dir}/best.pt')
                        prev_map = map_k
                        stop_cnt = 0
                    else :
                        print(f'{stop_cnt}: stop_cnt')
                        stop_cnt += 1
                    model.train()
