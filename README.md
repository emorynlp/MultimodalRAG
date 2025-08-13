# 🚙 Reference-Aligned Retrieval-Augmented Question Answering over Heterogeneous Proprietary Documents [[arXiv]](https://arxiv.org/pdf/2502.19596)

### ⚙️ Data Generation
How to convert the original PPT slides into a text corpus and extract question-answer pairs is as follows.
```
python slide_extraction.py --input_directory ./slide \
                           --output_directory ./md
```
- Please refer to the appendix of the [paper](https://arxiv.org/pdf/2502.19596) for the prompts used to extract question-answer pairs.

### ⚙️ RAG Components
How to train the <b>Retriever</b>, <b>Re-ranker</b>, and <b>Answer Generation Model</b>, as well as how to use the <b>Reference Matching Algorithm</b> with the Re-ranker, are as follows.

#### ✔ Retriever
- We fine-tuned [BGE-M3](https://huggingface.co/BAAI/bge-m3) with the publicly available [code](https://github.com/FlagOpen/FlagEmbedding).

#### ✔ Re-ranker
```
python reranker.py --train_data_path ./sample_data/reranker/train.csv \
                   --val_data_path ./sample_data/reranker/val.csv \
                   --from_pretrained BAAI/bge-m3 \
                   --epochs 10 \
                   --lr 1e-5 \
                   --negative_size 3 \
                   --token_k 3 \
                   --check_step 100 \
                   --save_step 1000 \
                   --quit_cnt 3 \
                   --save_dir ./reranker_trained
```

#### ✔ Answer Generation Model
```
python llm.py --train_data_path ./sample_data/answer_generation/train.csv \
              --val_data_path ./sample_data/answer_generation/val.csv \
              --from_pretrained Qwen/Qwen2.5-72B-Instruct \
              --epochs 3 \
              --lr 1e-5 \
              --logging_steps 10 \
              --save_dir ./qwen_trained \
              --log_dir ./log
```

#### ✔ Reference Matching Algorithm
```
python reference_matching.py --from_pretrained BAAI/bge-m3 \
                             --reranker_path ./reranker_trained/best.pt \
                             --save_path ./result_segments.json \
                             --question "Has there been an example where improving the vehicle structure prevented the fracture of components?" \
                             --generated_answer "Yes, there have been examples in several impact tests where improvements to the vehicle structure have prevented the fracture of components. For instance, in the 60kph Side Impact test, structural upgrades to the side door beams helped prevent component fractures. Additionally, the reinforced cross-members in the 80kph Front Impact test played a crucial role in reducing damage to the front frame, preventing fractures that usually occur at high speeds. These examples illustrate the effectiveness of vehicle design improvements in mitigating damage to critical components during collisions." \
                             --retrieved_chunks ["In the 60kph Side Impact test, structural improvements were made to the vehicle's side door beams to enhance safety during side collisions. These upgrades were specifically targeted to prevent fractures and deformation of key structural components, such as the door frame.",
                                                 "During the 80kph Front Impact test, the vehicle was redesigned to include reinforced cross-members in the front bumper area. This modification proved to be effective in reducing the severity of damage to the front frame, preventing fractures that typically occur at high speeds.",
                                                 "The addition of a more rigid roof structure was another major change. This improvement was especially beneficial in preventing the roof from collapsing during high-speed rollover accidents, where the roof integrity is often compromised.",
                                                 "For the latest prototype model, the addition of extra reinforcements in the rear underbody of the vehicle was designed to prevent the fracturing of key components during rear-end collisions. This structural improvement was tested in a series of crash simulations, where the results showed a marked reduction in damage compared to previous models.",
                                                 "A major change in the 120kph frontal crash tests involved the introduction of an advanced crumple zone system in the vehicle's front-end design. While the crumple zones absorbed much of the impact force, the newly reinforced side pillars helped prevent fractures of internal components like the engine block and steering mechanism."]
```

## 💬 Prompts

<details>
<summary>Prompt used to convert a presentation slide into markdown text</summary>
<img width="1000" alt="Screenshot 2025-05-21 at 5 07 37 PM" src="https://github.com/user-attachments/assets/007f3c2f-ec4b-4e1c-a686-5a4f570d0cab" />
</details>

<details>
<summary>Prompt used to generate Q&A pairs from a given chunk</summary>
<img width="1000" alt="Screenshot 2025-05-21 at 5 09 20 PM" src="https://github.com/user-attachments/assets/360f4c65-f094-47f1-9dae-96e2e8894f86" />
</details>

<details>
<summary>Prompt used to compare answers from LLM-only models</summary>
<img width="1000" alt="Screenshot 2025-05-21 at 5 09 52 PM" src="https://github.com/user-attachments/assets/da3f1805-58c9-4bd2-800f-13bafeeadaaa" />
</details>

<details>
<summary>Prompt used to compare answers from LLM-only and RAG models</summary>
<img width="1000" alt="Screenshot 2025-05-21 at 5 10 23 PM" src="https://github.com/user-attachments/assets/d44a3d38-81c6-4f16-b46d-ccfe8495d193" />
</details>

<details>
<summary>Prompt used for LLM reference matching</summary>
<img width="1000" alt="Screenshot 2025-05-21 at 5 10 52 PM" src="https://github.com/user-attachments/assets/90c978bc-fa08-441b-ae34-e18a31445cfb" />
</details>
