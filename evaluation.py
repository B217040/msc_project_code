#! /usr/bin/python

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from dataset_query import load_data
import torch
from evaluate import load

def load_model(model_size):
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}") #load whisper processer
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}").to('cuda')
    model.config.forced_decoder_ids = None # pairs of integers which indicates a mapping from generation indices to token indices
    return processor, model


def make_prediction(sample):
    audio = sample["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # return_tensors from WhisperFeatureExtractor class - pt is PyTorch format
    batch["reference"] = processor.tokenizer._normalize(batch['text'])


    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch


result = librispeech_test_clean.map(make_predicton) #map applies function to all samples in dataset
wer = load("wer")
print(100 * wer.compute(references=result["reference"], predictions=result["prediction"]))

if __name__ == '__main__':
    eval_data = load_dataset('cy_gb')['test']
    processor, model = load_model('tiny')