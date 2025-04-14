import gc
import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import argparse

def prepare_sample(sample):
    messages = [
        {"role": "system", "content": "Please provide an accurate answer to the question."},
        {"role": "user", "content": f"Question: {sample['question']}"},
        {"role": "assistant", "content": sample['answer']}
    ]
    return {"messages": messages}

def preprocess_function(examples):
    texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in examples["messages"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=2048)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--huggingface_token", type=str, default='', help='Enter your token for huggingface login')
    parser.add_argument("--train_data_path", type=str, default='./sample_data/answer_generation/train.csv')
    parser.add_argument("--val_data_path", type=str, default='./sample_data/answer_generation/val.csv')
    parser.add_argument("--from_pretrained", type=str, default='Qwen/Qwen2.5-72B-Instruct')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default='./qwen_trained')
    parser.add_argument("--log_dir", type=str, default='./log')
    args = parser.parse_args()

    login(token=args.huggingface_token)
    
    datasets = load_dataset('csv', data_files={
        'train': args.train_data_path,
        'validation': args.val_data_path
    })    
    train_dataset = datasets['train'].map(prepare_sample, remove_columns=datasets['train'].column_names)
    val_dataset = datasets['validation'].map(prepare_sample, remove_columns=datasets['validation'].column_names)
        
    model_id = args.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       attn_implementation="flash_attention_2"
    )    
    assert model.config.vocab_size == len(tokenizer), f"Model vocab size {model.config.vocab_size} does not match tokenizer length {len(tokenizer)}"
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_dir=args.log_dir,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        save_total_limit=1,
    )
    
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,  
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=2048
    )
    
    trainer.train()
