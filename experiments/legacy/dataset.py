# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:06:34 2024

@author: HarveyC
"""
from datasets import Dataset, DatasetDict, Audio
import os
import pandas as pd

audio_files = []
transcriptions = []
missing_transcriptions_files = []

directory = r"D:\fin"

root_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
batch_size = 2

num_iterations = (len(root_dirs) + batch_size - 1) // batch_size


for i in range(num_iterations):

    start_idx = i * batch_size
    end_idx = min(start_idx + batch_size, len(root_dirs))

    current_batch = root_dirs[start_idx:end_idx]
    # 遍歷每個根目錄
    for root_dir in current_batch:
        root_path = os.path.join(directory, root_dir)
        audio_files = []
        transcriptions = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                if file.endswith(".wav") and "_" in file:
                    audio_file_path = os.path.join(root, file)
                    audio_files.append(audio_file_path)
                    
                    base_name, number = file.rsplit('_', 1)
                    number = int(number.split('.')[0])
                    
                    csv_file_name = f"{base_name}.csv"
                    csv_file_path = os.path.join(os.path.dirname(root), csv_file_name)
                    
                    if os.path.exists(csv_file_path):
                        df = pd.read_csv(csv_file_path, encoding='big5')
                        if number <= len(df):
                            transcription = df.iloc[number - 1]['correct']
                            transcription = str(transcription).replace('\"', '').replace(',', '，')
                            transcriptions.append(transcription)
                        else:
                            transcriptions.append("")
                            missing_transcriptions_files.append(audio_file_path)
                    else:
                        transcriptions.append("")
                        missing_transcriptions_files.append(audio_file_path)
                        break
                
# In[3]:
data = {
    "audio": audio_files,
    "sentence": transcriptions
}

custom_dataset = Dataset.from_dict(data)

common_voice = DatasetDict()
split_dataset = custom_dataset.train_test_split(test_size=0.2)
common_voice["train"] = split_dataset["train"]
common_voice["test"] = split_dataset["test"]

print(common_voice)
print(common_voice["train"][0])

# In[]:
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
model_name_or_path = r'model_name_or_path'
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)

feature_extractor
# In[0]:
task = "transcribe"
language='zh'

tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# In[1]:
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
print(common_voice["train"])

# In[4]:
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
  
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
# In[6]:   
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map='auto')


# In[7]:
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

# In[8]:
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="reach-vb/test",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=100,
    # max_steps=100, # only for testing purposes, remove this from your final run :)
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# In[9]:
trainer.train()
# In[10]:
    
# 輸出權重
from peft import PeftModel
peft_model_id = r"peft_model_path"
model.save_pretrained(peft_model_id)
# In[merge]:

from transformers import WhisperModel
from peft import PeftModel
# In[merge]:
peft_model_id = r"peft_model_path"
base_model = WhisperModel.from_pretrained(model_name_or_path)
# 載入第一個參數檔案
lora_model = PeftModel.from_pretrained(base_model, peft_model_id,ignore_mismatched_keys=True)
merge_model = lora_model.merge_and_unload()

merge_model.save_pretrained('merge_model_path')

# In[]:
import gc
del model
torch.cuda.empty_cache()
gc.collect()

# In[test]:
import torch
from transformers import pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
 "automatic-speech-recognition",
  model=r'model_path',
  chunk_length_s=30,
  device=device,
)

audio=r"C:\Users\XX\Desktop\audio.wav"
prediction = pipe(audio, batch_size=8)["text"]
prediction = pipe(audio, batch_size=8, return_timestamps=True)["chunks"]

