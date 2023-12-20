from utils import create_labels_dict, create_labels,encode_batch
from transformers.utils.dummy_tf_objects import create_optimizer
from transformers import TrainingArguments
from transformers import AdapterTrainer, EvalPrediction
from transformers import Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback
import numpy as np
from utils import compute_metrics
import pandas as pd
from datasets import Dataset
from transformers.adapters import AutoAdapterModel 
from transformers import AutoTokenizer
 



if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/MARBERT')
    marbert = AutoAdapterModel.from_pretrained('UBC-NLP/MARBERT')

        
    train_new = pd.read_csv('/path/to/train/data',lineterminator='\n')
    dev_new = pd.read_csv("/path/to/dev/data", lineterminator='\n')
    test_new = pd.read_csv("/path/to/test/data",lineterminator='\n')
    
    train = Dataset.from_pandas(train_new)
    dev = Dataset.from_pandas(dev_new)
    test = Dataset.from_pandas(test_new)
        
    labels = list(set(list(train['dialect'])))
    labels_dict = create_labels_dict(labels)
    
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
    
    marbert.config.problem_type ='classification'
    marbert.add_adapter('adapter_name', set_active=True, overwrite_ok=True,)

    marbert.add_classification_head('adapter_head', len(labels_dict),
                                    overwrite_ok =True)
    marbert.set_active_adapters("adapter_name")
    print(marbert.adapter_summary(as_dict=True))
    
    
   

    training_args = TrainingArguments(
        num_train_epochs=25,
        learning_rate= 0.0001,
        save_total_limit = 45,
        lr_scheduler_type = 'cosine',
        warmup_steps = 25 * 15680 * 0.1,
        evaluation_strategy ='epoch',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        output_dir="/path/to/output/dir",
        overwrite_output_dir=True,
        load_best_model_at_end = True,
        remove_unused_columns=False,
        metric_for_best_model = 'f1',
        optim= 'adamw_torch',
        save_strategy = 'epoch',
    )


    trainer = AdapterTrainer(
        model=marbert,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_metrics,


        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    trainer.train()

