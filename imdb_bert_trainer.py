"""BERT-base fine-tuning — original trainer entry point, direct PyCharm run.

imdb_bert_native.py, imdb_bert_scratch.py and imdb_bert_trainer.py all load the
same pretrained bert-base-uncased checkpoint.  imdb_bert_scratch.py is
misnamed: it calls from_pretrained and is not a from-scratch run.  This trainer
is the single representative BERT run.
"""

from hf_support import run_hf

if __name__ == "__main__":
    run_hf(
        model_key="bert",
        display_name="BERT-base",
        pretrained_name="bert-base-uncased",
        source_file="imdb_bert_trainer.py",
        lr=2e-5,
        batch_size=8,
        eval_batch_size=16,
        max_epochs=2,
        patience=2,
        max_len=128,
        notes="Duplicate native/scratch/trainer entry points; representative BERT run.",
    )
