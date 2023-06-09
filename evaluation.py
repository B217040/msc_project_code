#! /usr/bin/python

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
from evaluate import load

def load_model(model_size):
    '''
    Loads model and processor
    '''
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}") #load whisper processer
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}").to('cuda')
    model.config.forced_decoder_ids = None # pairs of integers which indicates a mapping from generation indices to token indices
    return processor, model

def make_prediction(sample, processer, model):
    '''
    Processes sample audio
    Normalises transcription and adds as entry 'reference' to data dict
    Predicts audio transcription with model and adds as entry 'prediction'
    '''
    audio = sample["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # return_tensors comes from WhisperFeatureExtractor class -> pt is PyTorch format

    sample["reference"] = processor.tokenizer._normalize(sample['raw transcription'])

    with torch.no_grad(): #used for inference - no backward pass is called, saves memory
        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)
    sample["prediction"] = processor.tokenizer._normalize(transcription)

    return sample

def calculate_WER(eval_data, processer, model):
    '''
    Makes predictions on dataset
    Returns WER
    '''
    result = eval_data.map(make_predicton) #map applies function to all samples in dataset
    wer = load("wer")
    return 100 * wer.compute(references=result["reference"], predictions=result["prediction"])

if __name__ == '__main__':
    
    model_size = 'tiny'
    eval_data = load_from_disk("/work/tc046/tc046/pchamp/data/fleurs_welsh_test")

    processor, model = load_model(model_size)
    WER = calculate_WER(eval_data[0:3], processor, model)
    print('WER of whisper-{model_size} on Welsh FLEURS test set is {WER}')

