#### (🚧 Work in Progress)
## Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems [[Paper]](https://arxiv.org/pdf/2502.19596)

<img width="1000" alt="Screenshot 2025-04-14 at 11 44 13 AM" src="https://github.com/user-attachments/assets/8ace61fd-0946-4cca-b63d-262b2c7b75e8" />

### Data Generation
- TBU

### RAG Components
How to train the <b>Retriever</b>, <b>Re-ranker</b>, and <b>Answer Generation Model</b>, as well as how to use the <b>Reference Matching Algorithm</b> with the Re-ranker, are as follows.

#### ✔ Retrieval
- We fine-tuned [BGE-M3](https://huggingface.co/BAAI/bge-m3) with the publicly available [code](https://github.com/FlagOpen/FlagEmbedding).

#### ✔ Re-ranking
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

#### ✔ Answer Generation
```
```

#### ✔ Reference Matching 
```
python reference_matching.py --from_pretrained BAAI/bge-m3 \
                             --reranker_path ./reranker_trained/best.pt \
                             --save_path ./result_segments.json \
                             --question "Has there been an example where improving the vehicle structure prevented the fracture of components?" \
                             --retrieved_chunks ["In the 60kph Side Impact test, structural improvements were made to the vehicle's side door beams to enhance safety during side collisions. These upgrades were specifically targeted to prevent fractures and deformation of key structural components, such as the door frame.",
                                                 "During the 80kph Front Impact test, the vehicle was redesigned to include reinforced cross-members in the front bumper area. This modification proved to be effective in reducing the severity of damage to the front frame, preventing fractures that typically occur at high speeds.",
                                                 "The addition of a more rigid roof structure was another major change. This improvement was especially beneficial in preventing the roof from collapsing during high-speed rollover accidents, where the roof integrity is often compromised.",
                                                 "For the latest prototype model, the addition of extra reinforcements in the rear underbody of the vehicle was designed to prevent the fracturing of key components during rear-end collisions. This structural improvement was tested in a series of crash simulations, where the results showed a marked reduction in damage compared to previous models.",
                                                 "A major change in the 120kph frontal crash tests involved the introduction of an advanced crumple zone system in the vehicle's front-end design. While the crumple zones absorbed much of the impact force, the newly reinforced side pillars helped prevent fractures of internal components like the engine block and steering mechanism."]
```


    
