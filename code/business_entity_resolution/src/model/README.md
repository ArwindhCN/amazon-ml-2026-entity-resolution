# Models

Nothing in this folder is published: the trained models are built from the challenge data, and the
pretrained transformer is downloaded from Hugging Face.

| Path | What | How to get it |
|---|---|---|
| `pretrained/ms-marco-MiniLM-L-6-v2/` | Pretrained cross-encoder, Apache 2.0, 22.7M parameters | `hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir code/business_entity_resolution/src/model/pretrained/ms-marco-MiniLM-L-6-v2` (uses `huggingface_hub` from requirements.txt) |
| `translit.json` | Learned Indian-script → Latin word dictionary (1,305 words) | `learn_translit.py` |
| `v7c/` | Candidate filter + cascade model | `train_cascade.py` |
| `v8/` | Final model: filter, final LightGBM, decision settings, two fine-tuned cross-encoder folds | `train_ce.py` + `train_final.py` |

The full sequence of commands is in `../../README.md`.
