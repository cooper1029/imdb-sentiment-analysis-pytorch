"""DistilBERT-base fine-tuning — original trainer entry point, direct PyCharm run.

The native and trainer files were duplicate entry points for the same
pretrained checkpoint; this trainer is the representative run.  Runtime support
(checkpoint/resume, early stopping, metrics, real PyCharm screenshots) lives in
hf_support.py, while the model and tokenizer remain the original pretrained
DistilBERT-base.
"""

from hf_support import run_hf

if __name__ == "__main__":
    run_hf(
        model_key="distilbert",
        display_name="DistilBERT-base",
        pretrained_name="distilbert-base-uncased",
        source_file="imdb_distilbert_trainer.py",
        lr=5e-5,
        batch_size=16,
        eval_batch_size=32,
        max_epochs=3,
        patience=2,
        max_len=128,
        notes="Duplicate native/trainer entry points; this is the representative run.",
    )
