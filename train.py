from utils import create_labels_dict, create_labels,encode_batch
from transformers.utils.dummy_tf_objects import create_optimizer
from transformers import TrainingArguments
from transformers import AdapterTrainer, EvalPrediction
from transformers import Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback
import numpy as np
from utils import compute_metrics


best_f1 = 0

training_args = TrainingArguments(
    num_train_epochs=25,
    learning_rate= 0.0001,
    save_total_limit = 45,
    lr_scheduler_type = 'cosine',
    warmup_steps = 25 * 15680 * 0.1,
    evaluation_strategy ='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    output_dir="/content/drive/MyDrive/qadi9",
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

