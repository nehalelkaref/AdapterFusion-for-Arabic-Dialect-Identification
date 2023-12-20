
from utils import *
from transformers import TrainingArguments
from transformers import AdapterTrainer
from transformers import EarlyStoppingCallback
import numpy as np
from utils import compute_metrics
import pandas as pd
from datasets import Dataset
from transformers.adapters import AutoAdapterModel, Fuse
from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')
    marbert = AutoAdapterModel.from_pretrained('UBC-NLP/MARBERT')
    
    marbert.load_adapter("path/to/saved/adapter1", set_active=True, with_head=False)
    marbert.load_adapter("path/to/saved/adapter2", set_active=True, with_head=False)
    marbert.load_adapter("path/to/saved/adapter3", set_active=True, with_head=False)
    
    train_new = pd.read_csv('/path/to/train/data',lineterminator='\n')
    dev_new = pd.read_csv("/path/to/dev/data", lineterminator='\n')
    test_new = pd.read_csv("/path/to/test/data",lineterminator='\n')
    
    train = Dataset.from_pandas(train_new)
    dev = Dataset.from_pandas(dev_new)
    test = Dataset.from_pandas(test_new)
    
    
    labels = list(set(list(train['dialect'])))
    labels_dict = create_labels_dict(labels)
    
    adapter_setup= Fuse("adapter1","adapter2", "adapter3")
    marbert.add_adapter_fusion(adapter_setup  ,  overwrite_ok=True)
    marbert.set_active_adapters(adapter_setup)
    marbert.train_adapter_fusion(marbert.active_adapters)
    marbert.add_classification_head("adapter fusion",
                                    num_labels=len(labels_dict),
                                    overwrite_ok=True)
    
    labels_train = create_labels(list(train['dialect']), labels_dict)
    labels_dev = create_labels(list(dev['label']) ,labels_dict)
    labels_test = create_labels(list(test['dialect']) ,labels_dict)
    
    
    train = train.add_column('labels_train', labels_train)
    dev= dev.add_column('labels_dev', labels_dev)
    test= test.add_column('labels_test', labels_test)
    
    train = train.map(encode_batch, batched=True)
    dev = dev.map(encode_batch, batched=True)
    test = test.map(encode_batch, batched=True)

    train = train.rename_column("labels_train", "labels")
    dev = dev.rename_column("labels_dev", "labels")
    test = test.rename_column("labels_test", "labels")

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dev.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=20,
    save_total_limit = 45,
    lr_scheduler_type = 'cosine',
    warmup_steps = 20* (len(train)/32) * 0.1,
    evaluation_strategy ='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="aocbi",
    overwrite_output_dir=True,
    load_best_model_at_end = True,
    remove_unused_columns=True,
    metric_for_best_model = 'f1',
    optim = 'adamw_torch',
    save_strategy = 'epoch',
)
    
    trainer = AdapterTrainer(

    model=marbert,
    args=training_args,
    train_dataset=train,
    eval_dataset=dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)
    trainer.train()