from datasets import load_dataset
sampling_rate = 16000 # found in Fleurs paper

def load_data(lang, split_type, streaming=False):
    '''
    Load Fleurs dataset from huggingface
    :param link: dataset name
    :param lang: langauge in format '[lang]_[region]' - 'cy_gb' or 'ca_es'
    :param split_type: train, test, or dev
    :return: loaded dataset
    '''
    dataset = load_dataset("google/fleurs", lang, split=split_type)
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
    #print('downloading data...')

    #welsh_train_dataset = load_data("cy_gb", "train")
    #welsh_dev_dataset = load_data("cy_gb", "validation")
    welsh_test_dataset = load_data("cy_gb", "test")

    print(welsh_test_dataset[0:3])

    #train_stats = get_stats(welsh_train_dataset)
    #print('for TRAIN dataset...')
    #print(f'number of pairs = {train_stats[0]}, number of hours = {train_stats[1]}')

    #dev_stats = get_stats(welsh_dev_dataset)
    #print('for DEV dataset...')
    #print(f'number of pairs = {dev_stats[0]}, number of hours = {dev_stats[1]}')

    test_stats = get_stats(welsh_test_dataset)
    print('for TEST dataset...')
    print(f'number of pairs = {test_stats[0]}, number of hours = {test_stats[1]}')
    #print('hello world')
