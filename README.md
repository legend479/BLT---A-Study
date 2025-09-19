# BLT---A-Study

## Directory structure:
```
BLT---A-Study
├─ data/
│  ├─ train.csv         # Column format: "input","target" (target = reversed input)
│  └─ test.csv
├─ src/
│  ├─ patcher.py        # entropy patcher
│  ├─ tokenizer.py      # tiny char tokenizer
│  ├─ baseline_model.py # baseline model (no patching)
│  ├─ blt_model.py      # BLT_poc 
│  ├─ dataset.py        # dataset + collate + preprocessing pipeline
│  ├─ train.py          # simple train loop (train & eval)
│  ├─ infer.py          # interactive inference
│  └─ eval.py           # produce predictions CSVs + metrics
├─ predictions/
│  ├─ predictions_blt.csv
│  └─ predictions_baseline.csv
├─ docs/
│  ├─ REPORT.md
│  └─ LMA Assignment 1.pdf
├─ requirements.txt
└─ README.md
```
## PLAN:
