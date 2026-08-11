import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import pandas as pd
import numpy as np
from glob import glob

import whisper
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

import re
import cn2an
import cx_Oracle

import shutil
import subprocess

import datetime
today = datetime.date.today().strftime('%Y%m%d')

import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import warnings

warnings.filterwarnings('ignore')

from StarCC import PresetConversion
convert = PresetConversion(src='cn', dst='tw', with_phrase=True)

from noisereduce.torchgate import TorchGate as TG
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import librosa
import librosa.display
import soundfile as sf
import sys
from Portfolio.asr_nlp_pipeline.src.STT.correcting import *
import gc


# Log processing stages into log.txt
def work_log(log_message):
    log_path = os.path.join(log_folder, log_file)
    with open(log_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{current_time}] {log_message}\n')


# Log file processing start information
def item_log(file_name):
    log_path = os.path.join(log_folder, log_file)
    with open(log_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{current_time}] File: {file_name}, conversion started\n')


# Move file to "finished" folder
def move_to_finished_folder(file_name):
    source_path = os.path.join(data_folder, file_name)
    target_path = os.path.join(finished_folder, file_name)
    shutil.move(source_path, target_path)


# Audio denoising
def denoise(input_file):
    data, rate = librosa.load(input_file, sr=None)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    data = librosa.util.normalize(data)
    data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)

    # Minimum smoothing window (>=64 ms)
    min_time_mask_smooth_ms = max(64, int(1000 * len(data) / rate / 100))

    # TorchGate noise reduction
    tg = TG(
        sr=rate,
        nonstationary=True,
        n_fft=2048,
        prop_decrease=0.95,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=min_time_mask_smooth_ms
    ).to(device)

    enhanced_speech = tg(data_tensor)
    enhanced_speech_np = enhanced_speech.squeeze().cpu().numpy()
    enhanced_speech_np = librosa.util.normalize(enhanced_speech_np)

    # Save denoised audio (overwrite original)
    sf.write(input_file, enhanced_speech_np, rate)


# Folder paths
data_folder = r'E:\data_folder'
finished_folder = r'E:\finished_folder'
outputs_folder = r'E:\outputs_folder'
log_folder = r'E:\log_folder'
stage_folder = r'E:\stage_folder'
error_folder = r'E:\error_folder'

log_file = f'{today}log.txt'


# Create folders if not exist
for folder in [data_folder, finished_folder, outputs_folder, log_folder, stage_folder, error_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# Load models
work_log('Loading speaker diarization model...')
pipeline = Pipeline.from_pretrained("E:\\CS_STT\\config.yaml").to(torch.device("cuda:0"))
work_log('Speaker diarization model loaded.')

work_log('Loading STT model...')
model_path = r'C:\Users\XXXX\.cache\huggingface\hub\faster-whisper-large-v2'
model = WhisperModel(model_path, device="cuda", compute_type="float16")
work_log('STT model loaded.')


# Convert MP3 to WAV using ffmpeg
def mp3_to_wav(mp3_path):
    command = f'ffmpeg -i {mp3_path} {mp3_path[:-4]}.wav'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode("utf-8"), error


# Speech-to-text main function
def speach_to_text(log_file, audio_path):
    s_time = time.time()
    try:
        segments, info = model.transcribe(
            audio_path,
            condition_on_previous_text=False,
            initial_prompt='電話響鈴',
            language="zh"
        )

        df = pd.DataFrame([
            {'start': segment.start, 'end': segment.end, 'text': convert(segment.text)}
            for segment in segments
        ])

        file_name = os.path.basename(audio_path)
        name, _ = os.path.splitext(file_name)

        df.to_csv(os.path.join(stage_folder, name) + '.tsv', sep='\t', index=False)

        work_log(f'{audio_path} STT finished. Inference time: {time.time() - s_time}')

    except Exception as e:
        work_log(f'Error while processing file: {str(e)}')


# Speaker diarization
def sd(audio_path):
    s_time = time.time()

    dia = pipeline(audio_path, num_speakers=2)

    work_log(f'{audio_path} SD finished. Inference time: {time.time() - s_time}')

    file_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(file_name)

    tsv_path = os.path.join(stage_folder, name) + '.sd.tsv'

    with open(tsv_path, 'w') as f:
        f.write('start\tend\tspeaker\n')
        for speech_turn, track, speaker in dia.itertracks(yield_label=True):
            f.write(f"{round(speech_turn.start,2)}\t{round(speech_turn.end,2)}\t{speaker}\n")

    work_log(f'{audio_path} SD saved')


# Merge STT and speaker diarization outputs
def merge_sd_stt_outputs(audio_path):
    file_name = os.path.basename(audio_path)
    name, _ = os.path.splitext(file_name)

    df_sd = pd.read_csv(os.path.join(stage_folder, name + '.sd.tsv'), sep='\t')
    df_stt = pd.read_csv(os.path.join(stage_folder, name + '.tsv'), sep='\t')

    # Identify customer speaker (shorter speaking time)
    total_time = df_sd.groupby('speaker')['end'].sum() - df_sd.groupby('speaker')['start'].sum()
    min_speaker = total_time.idxmin()

    df_sd_c = df_sd[df_sd['speaker'] == min_speaker]
    df_sd_c = df_sd_c[(df_sd_c['end'] - df_sd_c['start']) >= 1]

    # Fix timestamp errors (start >= end)
    for i, row in df_stt.iterrows():
        if i == 0:
            continue
        if row['start'] >= row['end']:
            df_stt.loc[i, 'start'] = df_stt.loc[i - 1, 'end']

    # Merge STT with SD (tolerance = 1.5 sec)
    df_stt = df_stt.sort_values(['start'])
    df_sd_c = df_sd_c.sort_values(['start'])

    df_stt_m1 = pd.merge_asof(df_stt, df_sd_c, on='start', direction='nearest', tolerance=1.5)

    df_stt = df_stt.sort_values(['end'])
    df_sd_c = df_sd_c.sort_values(['end'])

    df_stt_m2 = pd.merge_asof(df_stt, df_sd_c, on='end', direction='nearest', tolerance=1.5)

    concat_df = pd.concat(
        [df_stt, df_stt_m1['speaker'].astype(str), df_stt_m2['speaker'].astype(str)],
        axis=1,
        ignore_index=True
    )

    concat_df.columns = ['start', 'end', 'text', 'sd_start', 'sd_end']

    # Speaker propagation logic
    concat_df['speaker_c'] = 0
    conti_index = 0
    max_conti_index = 5

    for i, row in concat_df.iterrows():
        if row['sd_start'].startswith('SPEAKER') or row['sd_end'].startswith('SPEAKER'):
            concat_df.at[i, 'speaker_c'] = 1
            conti_index = 0
        else:
            if conti_index < max_conti_index:
                if conti_index >= 1:
                    concat_df.at[i, 'speaker_c'] = 1
                    conti_index += 1
                else:
                    concat_df.at[i, 'speaker_c'] = 0
            else:
                concat_df.at[i, 'speaker_c'] = 0
                conti_index = 0

    # Replace labels with A (agent) / C (customer)
    concat_df['speaker_c'].replace({0: 'A', 1: 'C'}, inplace=True)
    concat_df.drop(['sd_start', 'sd_end'], axis=1, inplace=True)

    output_path = os.path.join(outputs_folder, name) + '.itg.csv'
    concat_df.to_csv(output_path, index=False, encoding='big5', errors='ignore')

    return concat_df


# File system event handler
class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        time.sleep(3)
        run()


# Main processing pipeline
def run():
    audio_files = glob(os.path.join(data_folder, '*.wav')) + \
                  glob(os.path.join(data_folder, '*.mp3'))

    work_log(f'Processing file list: {audio_files}')

    for audio_path in audio_files:
        try:
            # Convert MP3 to WAV if needed
            if audio_path.lower().endswith('mp3'):
                try:
                    if ' ' in audio_path:
                        new_path = audio_path.replace(' ', '_')
                        os.rename(audio_path, new_path)
                        audio_path = new_path

                    mp3_to_wav(audio_path)
                    work_log(f'{audio_path} converted to WAV')

                    shutil.move(audio_path, os.path.join(finished_folder, os.path.basename(audio_path)))
                    audio_path = audio_path[:-4] + '.wav'

                except Exception as e:
                    work_log(f'MP3 conversion failed: {e}')
                    shutil.move(audio_path, os.path.join(error_folder, os.path.basename(audio_path)))

            # Denoising
            try:
                denoise(audio_path)
                work_log(f'{audio_path} denoised')
            except:
                pass

            # Speech-to-text
            speach_to_text(model, audio_path)

            # Retry if hallucination detected
            i = 1
            while True:
                file_name = os.path.basename(audio_path)
                name, _ = os.path.splitext(file_name)

                df_stt = pd.read_csv(os.path.join(stage_folder, name + '.tsv'), sep='\t')
                df_stt = df_stt.dropna()
                df_stt = df_stt[(df_stt['end'] - df_stt['start']) >= 0.5]

                lastrows = df_stt['text'].tail()

                if not lastrows.astype(str).nunique() == 1:
                    break
                elif i == 3:
                    work_log(f'{name} STT failed after retries')
                    break
                else:
                    work_log(f'{name} retry STT attempt {i}')
                    speach_to_text(model, audio_path)
                    i += 1

            if i == 3:
                shutil.move(audio_path, os.path.join(error_folder, os.path.basename(audio_path)))
            else:
                sd(audio_path)
                merge_sd_stt_outputs(audio_path)

                shutil.move(audio_path, os.path.join(finished_folder, os.path.basename(audio_path)))

        except Exception as e:
            work_log(f'{audio_path} processing failed')
            work_log(f'Error: {str(e)}')
            shutil.move(audio_path, os.path.join(finished_folder, os.path.basename(audio_path)))


if __name__ == '__main__':
    work_log('STT service started.')

    run()

    event_handler = AudioHandler()
    observer = Observer()
    observer.schedule(event_handler, path=data_folder, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        # Release GPU memory
        del model
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        observer.stop()
        observer.join()