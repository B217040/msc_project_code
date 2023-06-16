print('importing load dataset')
from datasets import load_dataset
print('importing load from disk')
from datasets import load_from_disk
sampling_rate = 16000 # found in Fleurs paper

def load_data(lang, streaming=False):
    '''
    Load Fleurs dataset from huggingface
    :param link: dataset name
    :param lang: langauge in format '[lang]_[region]' - 'cy_gb' or 'ca_es'
    :return: loaded dataset
    '''
    dataset = load_dataset("google/fleurs", lang)
    return dataset


def get_stats(dataset):
    '''
    :param dataset: input Fleurs dataset
    :return: number of audio/transcription pairs, total hours of speech
    '''
    total_mins = 0
    number_pairs = 0

    for entry in dataset:
        number_pairs += 1
        length_minutes = entry['num_samples']/(sampling_rate*60)
        total_mins += length_minutes

    number_hours = total_mins/60

    return number_pairs, number_hours


if __name__ == '__main__':


    welsh_data_local = load_from_disk("/work/tc046/tc046/pchamp/data/fleurs_welsh_test")
    local_test_stats = get_stats(welsh_data_local)
    print('running stats with local test data...')
    print(f'number of pairs = {local_test_stats[0]}, number of hours = {local_test_stats[1]}')

    '''
    #welsh_data = load_data("cy_gb")

    #welsh_train_dataset = welsh_data['train']
    #welsh_dev_dataset = welsh_data['validation']
    #welsh_test_dataset = welsh_data['test']

    #train_stats = get_stats(welsh_train_dataset)
    #dev_stats = get_stats(welsh_dev_dataset)
    #test_stats = get_stats(welsh_test_dataset)

    
    with open('/work/tc046/tc046/pchamp/results/dataset_query_results', 'a') as f:
        f.write('for TRAIN dataset...')
        f.write(f'number of pairs = {train_stats[0]}, number of hours = {train_stats[1]}')
        f.write('\n for TEST dataset...')
        f.write(f'\n number of pairs = {test_stats[0]}, number of hours = {test_stats[1]}')
        f.write('\n for DEV dataset...')
        f.write(f'\n number of pairs = {dev_stats[0]}, number of hours = {dev_stats[1]}')
    '''