# python sentiment_bias.py --data eec --output results/eec.gender.aula --model bert --bias_type gender
import json
import argparse
import csv
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        choices=['cp', 'ss', 'eec'],
                        help='Path to evaluation dataset.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--bias_type', type=str, required=True)
    args = parser.parse_args()

    return args


def calculate_sentiment(model, token_ids):
    labels = ['Negative', 'Neutral', 'Positive']
    # encoded_text = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_text)
    output = model(token_ids)

    scores = output[0][0].cpu().detach().numpy()
    scores = softmax(scores)
    sentiment = list(zip(labels, scores))
    sentiment = sorted(sentiment, key=lambda x: x[1], reverse=True)
    pred = sentiment[0][0]

    if pred == 'Positive':
        sentiment_score = 1
    elif pred == 'Neutral':
        sentiment_score = 0
    if pred == 'Negative':
        sentiment_score = -1

    return sentiment_score


def load_tokenizer_and_model(args):
    '''
    Load tokenizer and model to evaluate.
    '''
    # load model and tokenizer
    model_name = args.model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model


def main(args):
    '''
    Evaluate the bias in masked language models.
    '''
    tokenizer, model = load_tokenizer_and_model(args)
    total_score = 0
    toward_adv_score = 0
    equal_count = 0
    diff_count = 0

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    count = defaultdict(int)
    scores = defaultdict(int)
    total_count = defaultdict(int)

    # with open(f'data/test.{args.data}.{args.bias_type}.pairs.json') as f:
    with open(f'data/{args.data}.{args.bias_type}.pairs.json') as f:
        inputs = json.load(f)
        total_num = len(inputs)
        for input in tqdm(inputs):
            emotion_type = input['emotion_type']
            total_count[emotion_type] += 1
            # if emotion_type == '':
            #     continue
            if args.bias_type == 'gender':
                adv_sent_type = 'male_sentence'
                disadv_sent_type = 'female_sentence'
            elif args.bias_type == 'race':
                adv_sent_type = 'eu_sentence'
                disadv_sent_type = 'aa_sentence'
            adv_sentence = input[adv_sent_type]
            adv_token_ids = tokenizer.encode(adv_sentence, return_tensors='pt')
            disadv_sentence = input[disadv_sent_type]
            disadv_token_ids = tokenizer.encode(disadv_sentence, return_tensors='pt')

            with torch.no_grad():
                adv_score = calculate_sentiment(model, adv_token_ids)
                disadv_score = calculate_sentiment(model, disadv_token_ids)

            if adv_score == disadv_score:
                equal_count += 1
                scores[emotion_type] += 0
                continue
            if adv_score > disadv_score:
                toward_adv_score += 1
                scores[emotion_type] += 1
                diff_count += 1
            elif disadv_score > adv_score:
                diff_count += 1

            total_score += 1
            count[emotion_type] += 1
    sum_bias_score = 0        
    print(f'Equal count: {equal_count}, Diff_count: {diff_count}, Total count: {total_num}')
    fw = open(args.output, 'w')
    bias_score = round((toward_adv_score / total_score) * 100, 2)
    print('Bias score:', bias_score)
    fw.write(f'Bias score: {bias_score}\n')
    for emotion_type, score in scores.items():
        if score == 0:
            bias_score = 50
        else:
            bias_score = round((score / count[emotion_type]) * 100, 2)
        print(emotion_type, bias_score)
        fw.write(f'{emotion_type}: {bias_score}\n')
        sum_bias_score += count[emotion_type]


if __name__ == "__main__":
    args = parse_args()
    main(args)
