"""RoBERTa-base fine-tuning — original trainer entry point, direct PyCharm run.

Source used lr=5e-6 and batch size 4.  The managed direct run uses lr=1e-5 and
batch size 8 so it can finish on CPU while keeping conservative fine-tuning;
the pretrained model and tokenizer are unchanged.
"""

from hf_support import run_hf

if __name__ == "__main__":
    run_hf(
        model_key="roberta",
        display_name="RoBERTa-base",
        pretrained_name="roberta-base",
        source_file="imdb_roberta_trainer.py",
        lr=1e-5,
        batch_size=8,
        eval_batch_size=16,
        max_epochs=2,
        patience=2,
        max_len=64,
        notes="Source lr=5e-6, batch=4; CPU run uses lr=1e-5, batch=8.",
    )
