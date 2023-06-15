#! /usr/bin/python
print('importing transformers')
from transformers import WhisperProcessor, WhisperForConditionalGeneration
#from evaluate import load

def save_model_processor(model_size):
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
    model.config.forced_decoder_ids = None # pairs of integers which indicates a mapping from generation indices to token indices
    model.save_pretrained(f'/work/tc046/tc046/pchamp/model/whisper_model_{model_size}')

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}") #load whisper processer
    processor.save_pretrained(f'/work/tc046/tc046/pchamp/model/whisper_processor_{model_size}')

#def save_evaluator():
#    wer_evaluator = load("wer")
#    wer_evaluator.save('/work/tc046/tc046/pchamp/model/wer_evaluator')

if __name__ == '__main__':
    print('loading model and processor...')
    save_model_processor('tiny')

    #save_evaluator()
    #print('loading evaluator complete.')

