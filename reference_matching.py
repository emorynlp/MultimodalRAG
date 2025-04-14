import os
import json
import argparse

import numpy as np

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

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

def reference_matching(question, chunks, answer):
    sentences = answer.split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]        
    num_sentences = len(sentences)
    
    # Step 1
    score_matrix = np.full((num_sentences, num_sentences, len(chunks)), -np.inf)
    
    for start in range(num_sentences):
        for end in range(start, num_sentences):  # No max_segment constraint
            segment = " ".join(sentences[start:end + 1]).strip()
            for chunk_idx, chunk in enumerate(chunks):
                text = f'{segment} {tokenizer.sep_token} {chunk}'
                batch = tokenizer.batch_encode_plus([text],
                                                    max_length=4096 * 2,
                                                    padding='longest',
                                                    truncation=True,
                                                    return_tensors="pt")
                input_ids = batch['input_ids'].cuda()
                with torch.no_grad():
                    output = model(input_ids=input_ids, token_selected=False)['logits']
                    score_matrix[start, end, chunk_idx] = float(output.detach().cpu().numpy()[0][0])
    
    # Step 2
    dp = [-np.inf] * (num_sentences + 1)
    choice = [None] * (num_sentences + 1)
    dp[0] = 0  # Base case: no sentences processed
    
    for i in range(num_sentences):
        for j in range(i, num_sentences):
            for chunk_idx in range(len(chunks)):
                score = score_matrix[i, j, chunk_idx]
                if dp[j + 1] < dp[i] + score:
                    dp[j + 1] = dp[i] + score
                    choice[j + 1] = (i, j, chunk_idx)
    
    # Step 3
    result_segments = []
    current = num_sentences
    while current > 0:
        i, j, chunk_idx = choice[current]
        segment = " ".join(sentences[i:j + 1]).strip()
        result_segments.append((segment, chunks[chunk_idx], dp[current] - dp[i], list(range(i+1,j+2))))
        current = i
    
    result_segments.reverse()
    
    # Collect results
    answer_segments = {}
    for idx_, (segment, chunk, score, sentence_index) in enumerate(result_segments):
        print(f"Segment [{idx_}] '{segment}' \n-> Assigned to: '{chunks.index(chunk)}' with score: {score:.2f}\n")
        answer_segments[idx_] = {'segment': segment, 'sentence_index': sentence_index, 'chunk_id': chunks.index(chunk), 'score': float(score)}
    return answer_segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_pretrained", type=str, default='BAAI/bge-m3', help='model_name or local path')
    parser.add_argument("--reranker_path", type=str, default='./reranker_trained/best.pt')
    parser.add_argument("--save_path", type=str, default='./result_segments.json')
    parser.add_argument("--question", type=str, default='Has there been an example where improving the vehicle structure prevented the fracture of components?')
    parser.add_argument("--retrieved_chunks", type=list, default=["In the 60kph Side Impact test, structural improvements were made to the vehicle's side door beams to enhance safety during side collisions. These upgrades were specifically targeted to prevent fractures and deformation of key structural components, such as the door frame.",
 'During the 80kph Front Impact test, the vehicle was redesigned to include reinforced cross-members in the front bumper area. This modification proved to be effective in reducing the severity of damage to the front frame, preventing fractures that typically occur at high speeds.',
 'The addition of a more rigid roof structure was another major change. This improvement was especially beneficial in preventing the roof from collapsing during high-speed rollover accidents, where the roof integrity is often compromised.',
 'For the latest prototype model, the addition of extra reinforcements in the rear underbody of the vehicle was designed to prevent the fracturing of key components during rear-end collisions. This structural improvement was tested in a series of crash simulations, where the results showed a marked reduction in damage compared to previous models.',
 "A major change in the 120kph frontal crash tests involved the introduction of an advanced crumple zone system in the vehicle's front-end design. While the crumple zones absorbed much of the impact force, the newly reinforced side pillars helped prevent fractures of internal components like the engine block and steering mechanism."])
    parser.add_argument("--generated_answer", type=str, default='Yes, there have been examples in several impact tests where improvements to the vehicle structure have prevented the fracture of components. For instance, in the 60kph Side Impact test, structural upgrades to the side door beams helped prevent component fractures. Additionally, the reinforced cross-members in the 80kph Front Impact test played a crucial role in reducing damage to the front frame, preventing fractures that usually occur at high speeds. These examples illustrate the effectiveness of vehicle design improvements in mitigating damage to critical components during collisions.')
    args = parser.parse_args()
    
    # Load fine-tuned re-ranker
    tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    model = CustomAutoModelForSequenceClassification(model_name=args.from_pretrained, num_labels=1)
    model.load_state_dict(torch.load(args.reranker_path))
    model.cuda()
    model.eval()

    # Reference matching
    result_segments = reference_matching(args.question, args.retrieved_chunks, args.generated_answer)

    # Save the result
    with open(args.save_path, 'w') as f:
        json.dump(result_segments, f, indent=4)
