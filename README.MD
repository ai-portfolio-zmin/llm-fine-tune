LLM Fine-Tuning Platform (qLoRA, Hugging Face)

This repository implements a general fine-tuning and evaluation pipeline for
generation-based decision models using Hugging Face and LoRA.

The pipeline is designed to support tasks where a language model is trained to
produce deterministic, structured outputs (e.g. JSON) that represent discrete
decisions.  

The current implementation includes a retrieval router as the first
concrete use case.

Scope & Design

The repository is structured as a task-agnostic platform with a clear
separation between:

- Core training / evaluation logic (task-independent)
- Task-specific definitions (datasets, prompts, evaluation rules)

This allows new decision tasks to be added with minimal changes to the core
pipeline.[README.md](..%2Fagentflow%2FREADME.md)

At present, the platform supports:
- LoRA adapter fine-tuning
- Deterministic generation-based inference
- Structured output extraction and validation
- Balanced dataset evaluation with confusion matrices and aggregate metrics


Current Use Case: Retrieval Router

The initial task implemented in this platform is a 3-way semantic router that
classifies user queries with features and preview retrival scores into one of the following retrieval strategies:

- `bm25`
- `dense`
- `hybrid`

The model is trained to generate a strict JSON object of the form:


`{"route": "<bm25|dense|hybrid>"}`

Rather than using a classification head, routing is framed as a
generation-based decision task, ensuring compatibility with downstream
LLM-driven systems.

Model & Training Setup
Base model: microsoft/phi-3-mini-4k-instruct

Fine-tuning method: LoRA (adapter-only)

Decoding: deterministic (no sampling, temperature = 0)

Artifacts: only LoRA adapters are saved and versioned

Base model weights are loaded from Hugging Face at runtime and are not stored in
this repository.

Dataset & Splits
All datasets are balanced across classes.
```
Split	Size	Distribution
Train	1500	500 / 500 / 500
Validation	300	100 / 100 / 100
Test	300	100 / 100 / 100
```

Validation set is used only for model selection

Test set is used once for final reporting

Results Summary
Validation Performance (Model Selection)
```
Epoch	Accuracy	Macro-F1
1	0.437	0.438
3	0.547	0.522
5	0.587	0.584
```

Epoch 5 was selected as the final model.

Final Test Results (Epoch 5)
The final model was evaluated once on a held-out balanced test set.
```
Accuracy: 0.553
Macro-F1: 0.552
Recall: 0.553
```

Confusion Matrix (Test Set)
Rows correspond to true labels, columns to predicted labels.
```
True \ Pred	bm25	dense	hybrid
bm25	         53	  17	  30
dense	         16	  43	  41
hybrid	         5	  25	  70
```

Output Reliability
Generation is fully deterministic

JSON output is consistently extractable

No schema violations observed during evaluation

Fine-tuning enforces output structure beyond prompt-only approaches.

Repository Structure
```
llm-tuning/
├── config/                 # Training & evaluation configs
├── data/                   # Train / val / test datasets
├── hf_cache/               # Hugging Face cache 
├── model/                  # Saved LoRA adapters
├── output/                 # Evaluation outputs
├── src/
│   ├── core/               # Task-agnostic training & evaluation
│   │   ├── train_engine.py
│   │   ├── eval_engine.py
│   │   └── util.py
│   ├── tasks/              # Task-specific implementations
│   │   └── router/
│   │       ├── dataset.py
│   │       ├── prompt.py
│   │       └── eval.py
│   ├── logger_util.py
│   └── path_util.py
├── data-gen.ipynb          # Dataset generation notebook
├── requirements.txt
└── README.md
```

Training
```
python -m src.core.train_engine \
  --config config/train_router.yml \
  --train_set train \
  --num_train_epochs 5 \
  --learning_rate 1e-4
```

The training engine:

- loads the base model

- applies LoRA adapters

- trains on the specified split

- saves adapter weights only

Evaluation
Validation:

```
python -m src.core.eval_engine \
  --config config/train_router.yml \
  --eval_set val
 ```
  
Final test (run once):
```
python -m src.core.eval_engine \
  --config config/train_router.yml \
  --eval_set test
```

Evaluation reports:

- confusion matrix

- accuracy

- macro-F1

- recall

Extending the Platform
New tasks can be added by:

- Creating a new task directory under src/tasks/

- Defining task-specific dataset, prompt, and evaluation logic

- Reusing the existing training and evaluation engines

The platform is suitable for other generation-based decision tasks, such as:

- intent classification

- query or tool routing

- policy or rule selection

- schema-constrained JSON generation

Notes:

This project prioritizes engineering discipline and reproducibility

Test data is never used for training or tuning

Further gains are expected from improved data or labeling, not longer training