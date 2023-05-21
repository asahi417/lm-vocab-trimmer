# python preprocess.py --output data/eec.gender.pairs.json --bias_type gender
import json
import argparse
import csv
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bias_type', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    return args


def preprocess_instances(path):
    '''
    Extract stereotypical and anti-stereotypical sentences from crows-paris.
    '''
    data = []

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    for row in rows:
        example = {}
        example['bias_type'] = args.bias_type
        example['emotion_type'] = row['Emotion']
        if example['emotion_type'] == '':       # filter out instances that do not contain emotion_type
            # continue
            example['emotion_type'] = 'neutral'
        if args.bias_type == 'gender':
            if row['Gender'] == 'male':
                example['male_sentence'] = row['Sentence']
                template = row['Template']
                emotion_word = row['Emotion word']

                for row2 in rows:      
                    if row2['Gender'] == 'female':
                        if row2['Template'] == template and row2['Emotion word'] == emotion_word:
                            example['female_sentence'] = row2['Sentence']
                            temp = copy.deepcopy(example)
                            data.append(temp)

        elif args.bias_type == 'race':
            if row['Race'] == 'European':
                example['eu_sentence'] = row['Sentence']
                template = row['Template']
                emotion_word = row['Emotion word']

                for row2 in rows:      
                    if row2['Race'] == 'African-American':
                        if row2['Template'] == template and row2['Emotion word'] == emotion_word:
                            example['aa_sentence'] = row2['Sentence']
                            temp = copy.deepcopy(example)
                            data.append(temp)

    return data


def main(args):

    path = 'data/Equity-Evaluation-Corpus.csv'
    data = preprocess_instances(path)
  
    with open(args.output, 'w') as fw:
        json.dump(data, fw, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)