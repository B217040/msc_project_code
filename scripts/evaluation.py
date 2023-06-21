#! /usr/bin/python

print('starting import')
from transformers import WhisperProcessor, WhisperForConditionalGeneration
print('import transformers')
from datasets import load_from_disk
print('import datasets')
import torch
print('import torch')
from evaluate import load
print('import evaluate')

def load_model(model_size):
    '''
    Loads model and processor
    '''
    processor = WhisperProcessor.from_pretrained(f'/work/tc046/tc046/pchamp/model/whisper_processor_{model_size}') #load whisper processer
    model = WhisperForConditionalGeneration.from_pretrained(f'/work/tc046/tc046/pchamp/model/whisper_model_{model_size}')
    return processor, model

def make_prediction(batch):
    '''
    Processes sample audio
    Normalises transcription and adds as entry 'reference' to data dict
    Predicts audio transcription with model and adds as entry 'prediction'
    '''

    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # return_tensors comes from WhisperFeatureExtractor class -> pt is PyTorch format

    batch["reference"] = processor.tokenizer._normalize(batch['raw_transcription'])

    with torch.no_grad(): #used for inference - no backward pass is called, saves memory
        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch

def calculate_WER(result):
    '''
    Makes predictions on dataset
    Returns WER
    '''
    wer = load("/work/tc046/tc046/pchamp/msc_project_code/scripts/wer.py")
    return 100 * wer.compute(references=result["reference"], predictions=result["prediction"])

if __name__ == '__main__':

    model_size = 'tiny'

    eval_data = load_from_disk("/work/tc046/tc046/pchamp/data/fleurs_welsh_test")
    print('loaded data')
    processor, model = load_model(model_size)
    print('loaded model')
    result = eval_data.map(make_prediction) #map applies function to all samples in dataset
    print('result calculated')
    WER = calculate_WER(result)

    with open(f'/work/tc046/tc046/pchamp/results/whisper-{model_size}-results.txt', 'a') as f:
        f.write('WER of whisper-{model_size} on Welsh FLEURS test set is {WER}')