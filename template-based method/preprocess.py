import pandas as pd
import csv
import random
import math
import numpy as np
import json


def generate_ngrams(sentence, n):
    """
        Extract n-gram for a sentence.
        Args:
            sentence: the sentence waiting to generate.
            n: the number of continuous gram.
        Returns:
            ngrams: a list of n-gram for a sentence.
        """  # noqa: ignore flake8"
    words = sentence.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)
    return ngrams


def generate_all_ngrams(sentence):
    """
        Extract all n-gram for a sentence.
        Args:
            sentence: the sentence waiting to generate.
        Returns:
            all_ngrams: a list of all n-gram for a sentence.
        """  # noqa: ignore flake8"
    words = sentence.split()
    n = min(10, len(words))
    all_ngrams = []

    for i in range(1, n + 1):
        ngrams = generate_ngrams(sentence, i)
        all_ngrams.extend(ngrams)

    return all_ngrams


def csv2txt(input_path, output_path):
    """
        Convert CSV format data to txt dormat data.
        Args:
            input_path: the path to the input file.
            output_path: the path to the output file.
        """  # noqa: ignore flake8"
    with open(output_path, "w") as output_file:
        train_data = pd.read_csv(input_path, sep=',').values.tolist()
        success_sents = []
        for s in train_data:
            if s[0] not in success_sents:
                if len(success_sents) != 0:
                    output_file.write('\n')
                success_sents.append(s[0])
            if s[2] == 'o':
                new_label = s[2].upper()
            else:
                new_label = s[2]
            new_line = str(s[1]) + ' ' + new_label + '\n'
            output_file.write(new_line)

def json2template(input_path, output_path, dataset='multiconer', generate_negative=False):
    """
    Convert JSON format data to CSV format data with template.
    Args:
        input_path: the path to the input file.
        output_path: the path to the output file.
        dataset: the dataset to generate template.
        generate_negative: a boolean variable to specify whether to generate negative template.
    """
    output_filename = output_path

    if dataset == 'multiconer':
        extend_label = {
            'LOC': 'location',
            'PER': 'person',
            'CORP': 'corporation',
            'GRP': 'group',
            'PROD': 'product',
            'CW': 'creative work'
        }
    elif dataset == 'n2c2':
        extend_label = {
            'm': 'medicine',
            'do': 'dosage',
            'mo': 'mode',
            'f': 'frequency',
            'du': 'duration',
            'r': 'reason'
        }
    elif dataset == 'MIT_Movie':
        extend_label = {
            'ACTOR': 'actor',
            'YEAR': 'year',
            'TITLE': 'title',
            'GENRE': 'genre',
            'DIRECTOR': 'director',
            'SONG': 'song',
            'PLOT': 'plot',
            'REVIEW': 'review',
            'CHARACTER': 'character',
            'RATING': 'rating',
            'RATINGS_AVERAGE': 'ratings_average',
            'TRAILER': 'trailer'
        }

    with open(input_path, 'r',  encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_filename, "w", newline='',  encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        # Write header row
        csv_writer.writerow(["Source sentence", "Answer sentence"])

        for item in data:
            tokens = item['text']
            labels = item['label']

            buffer_token = ""
            buffer_label = ""
            positive_example = 0
            first = " ".join(tokens)
            all_ngrams = generate_all_ngrams(first)
            positive_grams = []

            for l, t in zip(labels, tokens):
                if l.split("-")[0] != 'I' and buffer_token != "":
                    answer_sentence = '%s is an %s entity' % (
                        buffer_token, buffer_label) if (buffer_label.lower().startswith(
                        "o") or buffer_label.lower().startswith(
                        "a")) else '%s is a %s entity' % (
                        buffer_token, buffer_label)
                    positive_example += 1
                    positive_grams += generate_all_ngrams(buffer_token)
                    csv_writer.writerow([first, answer_sentence])
                    buffer_token = ""
                    buffer_label = ""
                if l.split("-")[0] == 'B':
                    buffer_token = t
                    buffer_label = extend_label[l.split("-")[1]]
                if l.split("-")[0] == 'I':
                    buffer_token += " " + t
            if buffer_token != "":
                answer_sentence = '%s is an %s entity' % (
                    buffer_token, buffer_label) if (buffer_label.lower().startswith(
                    "o") or buffer_label.lower().startswith(
                    "a")) else '%s is a %s entity' % (
                    buffer_token, buffer_label)
                csv_writer.writerow([first, answer_sentence])
                positive_example += 1
                positive_grams += generate_all_ngrams(buffer_token)

            if generate_negative:
                negative_grams = [item for item in all_ngrams if item not in positive_grams]

                # Separate negative_grams by length
                negative_grams_by_length = {}
                for item in negative_grams:
                    length = len(item.split())
                    if length not in negative_grams_by_length:
                        negative_grams_by_length[length] = []
                    negative_grams_by_length[length].append(item)

                # Define the ratio for each length, including lengths greater than 4
                length_ratios = {
                    1: 0.648,
                    2: 0.30,
                    3: 0.037,
                    4: 0.011,
                    5: 0.003
                }
                # Assign the remaining probability (0.02) to lengths greater than 4
                remaining_ratio = 0.001

                # Create a list of all negative_grams and their probabilities
                all_negative_grams = []
                all_probabilities = []
                for length, ngrams in negative_grams_by_length.items():
                    all_negative_grams.extend(ngrams)
                    if length in length_ratios:
                        all_probabilities.extend([length_ratios[length]] * len(ngrams))
                    else:
                        all_probabilities.extend([remaining_ratio / len(negative_grams_by_length)] * len(ngrams))

                # Normalize the probabilities
                all_probabilities = np.array(all_probabilities)
                all_probabilities /= all_probabilities.sum()

                # Weighted random choice function
                def weighted_random_choice(items, weights, k):
                    return np.random.choice(items, size=k, replace=False, p=weights)

                # Calculate the number of samples
                negative_example = math.ceil(positive_example * 1.5)

                # Sample negative_grams based on the probability distribution if the list is not empty
                if all_negative_grams:
                    k = min(negative_example, len(all_negative_grams))
                    selected_negative_grams = weighted_random_choice(all_negative_grams, all_probabilities, k).tolist()

                    for g in selected_negative_grams:
                        answer_sentence = '%s is not a named entity' % g
                        csv_writer.writerow([first, answer_sentence])

def json2template_bangla(input_path, output_path, dataset='multiconer', generate_negative=False):
    """
    Convert JSON format data to CSV format data with template.
    Args:
        input_path: the path to the input file.
        output_path: the path to the output file.
        dataset: the dataset to generate template.
        generate_negative: a boolean variable to specify whether to generate negative template.
    """
    output_filename = output_path

    if dataset == 'multiconer':
        extend_label = {
            'LOC': 'স্থান',
            'PER': 'ব্যক্তি',
            'CORP': 'কর্পোরেশন',
            'GRP': 'দল',
            'PROD': 'পণ্য',
            'CW': 'সৃজনশীল কাজ'
        }
    elif dataset == 'n2c2':
        extend_label = {
            'm': 'medicine',
            'do': 'dosage',
            'mo': 'mode',
            'f': 'frequency',
            'du': 'duration',
            'r': 'reason'
        }
    elif dataset == 'MIT_Movie':
        extend_label = {
            'ACTOR': 'actor',
            'YEAR': 'year',
            'TITLE': 'title',
            'GENRE': 'genre',
            'DIRECTOR': 'director',
            'SONG': 'song',
            'PLOT': 'plot',
            'REVIEW': 'review',
            'CHARACTER': 'character',
            'RATING': 'rating',
            'RATINGS_AVERAGE': 'ratings_average',
            'TRAILER': 'trailer'
        }

    with open(input_path, 'r',  encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_filename, "w", newline='',  encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        # Write header row
        csv_writer.writerow(["Source sentence", "Answer sentence"])

        for item in data:
            tokens = item['text']
            labels = item['label']

            buffer_token = ""
            buffer_label = ""
            positive_example = 0
            first = " ".join(tokens)
            all_ngrams = generate_all_ngrams(first)
            positive_grams = []

            for l, t in zip(labels, tokens):
                if l.split("-")[0] != 'I' and buffer_token != "":
                    answer_sentence = '%s একটি %s সত্তা' % (buffer_token, buffer_label)
                    positive_example += 1
                    positive_grams += generate_all_ngrams(buffer_token)
                    csv_writer.writerow([first, answer_sentence])
                    buffer_token = ""
                    buffer_label = ""
                if l.split("-")[0] == 'B':
                    buffer_token = t
                    buffer_label = extend_label[l.split("-")[1]]
                if l.split("-")[0] == 'I':
                    buffer_token += " " + t
            if buffer_token != "":
                answer_sentence = '%s একটি %s সত্তা' % (buffer_token, buffer_label)
                csv_writer.writerow([first, answer_sentence])
                positive_example += 1
                positive_grams += generate_all_ngrams(buffer_token)

            if generate_negative:
                negative_grams = [item for item in all_ngrams if item not in positive_grams]

                # Separate negative_grams by length
                negative_grams_by_length = {}
                for item in negative_grams:
                    length = len(item.split())
                    if length not in negative_grams_by_length:
                        negative_grams_by_length[length] = []
                    negative_grams_by_length[length].append(item)

                # Define the ratio for each length, including lengths greater than 4
                length_ratios = {
                    1: 0.648,
                    2: 0.30,
                    3: 0.037,
                    4: 0.011,
                    5: 0.003
                }
                # Assign the remaining probability (0.02) to lengths greater than 4
                remaining_ratio = 0.001

                # Create a list of all negative_grams and their probabilities
                all_negative_grams = []
                all_probabilities = []
                for length, ngrams in negative_grams_by_length.items():
                    all_negative_grams.extend(ngrams)
                    if length in length_ratios:
                        all_probabilities.extend([length_ratios[length]] * len(ngrams))
                    else:
                        all_probabilities.extend([remaining_ratio / len(negative_grams_by_length)] * len(ngrams))

                # Normalize the probabilities
                all_probabilities = np.array(all_probabilities)
                all_probabilities /= all_probabilities.sum()

                # Weighted random choice function
                def weighted_random_choice(items, weights, k):
                    return np.random.choice(items, size=k, replace=False, p=weights)

                # Calculate the number of samples
                negative_example = math.ceil(positive_example * 1.5)

                # Sample negative_grams based on the probability distribution if the list is not empty
                if all_negative_grams:
                    k = min(negative_example, len(all_negative_grams))
                    selected_negative_grams = weighted_random_choice(all_negative_grams, all_probabilities, k).tolist()

                    for g in selected_negative_grams:
                        answer_sentence = '%s একটি নামকরণ করা সত্তা নয়' % g
                        csv_writer.writerow([first, answer_sentence])

def json2template_chinese(input_path, output_path, dataset='multiconer', generate_negative=False):
    """
    Convert JSON format data to CSV format data with template.
    Args:
        input_path: the path to the input file.
        output_path: the path to the output file.
        dataset: the dataset to generate template.
        generate_negative: a boolean variable to specify whether to generate negative template.
    """
    output_filename = output_path

    if dataset == 'multiconer':
        extend_label = {
            'LOC': '地点',  # Place/Location
            'PER': '人名',  # Person
            'CORP': '公司',  # Corporation
            'GRP': '群体',  # Group
            'PROD': '产品',  # Product
            'CW': '创作作品'  # Creative Work
        }
    elif dataset == 'n2c2':
        extend_label = {
            'm': 'medicine',
            'do': 'dosage',
            'mo': 'mode',
            'f': 'frequency',
            'du': 'duration',
            'r': 'reason'
        }
    elif dataset == 'MIT_Movie':
        extend_label = {
            'ACTOR': 'actor',
            'YEAR': 'year',
            'TITLE': 'title',
            'GENRE': 'genre',
            'DIRECTOR': 'director',
            'SONG': 'song',
            'PLOT': 'plot',
            'REVIEW': 'review',
            'CHARACTER': 'character',
            'RATING': 'rating',
            'RATINGS_AVERAGE': 'ratings_average',
            'TRAILER': 'trailer'
        }

    with open(input_path, 'r',  encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_filename, "w", newline='',  encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        # Write header row
        csv_writer.writerow(["Source sentence", "Answer sentence"])

        for item in data:
            tokens = item['text']
            labels = item['label']

            buffer_token = ""
            buffer_label = ""
            positive_example = 0
            first = " ".join(tokens)
            all_ngrams = generate_all_ngrams(first)
            positive_grams = []

            for l, t in zip(labels, tokens):
                if l.split("-")[0] != 'I' and buffer_token != "":
                    answer_sentence = '%s 是一个 %s 实体' % (buffer_token, buffer_label)
                    positive_example += 1
                    positive_grams += generate_all_ngrams(buffer_token)
                    csv_writer.writerow([first, answer_sentence])
                    buffer_token = ""
                    buffer_label = ""
                if l.split("-")[0] == 'B':
                    buffer_token = t
                    buffer_label = extend_label[l.split("-")[1]]
                if l.split("-")[0] == 'I':
                    buffer_token += " " + t
            if buffer_token != "":
                answer_sentence = '%s 是一个 %s 实体' % (buffer_token, buffer_label)
                csv_writer.writerow([first, answer_sentence])
                positive_example += 1
                positive_grams += generate_all_ngrams(buffer_token)

            if generate_negative:
                negative_grams = [item for item in all_ngrams if item not in positive_grams]

                # Separate negative_grams by length
                negative_grams_by_length = {}
                for item in negative_grams:
                    length = len(item.split())
                    if length not in negative_grams_by_length:
                        negative_grams_by_length[length] = []
                    negative_grams_by_length[length].append(item)

                # Define the ratio for each length, including lengths greater than 4
                length_ratios = {
                    1: 0.648,
                    2: 0.30,
                    3: 0.037,
                    4: 0.011,
                    5: 0.003
                }
                # Assign the remaining probability (0.02) to lengths greater than 4
                remaining_ratio = 0.001

                # Create a list of all negative_grams and their probabilities
                all_negative_grams = []
                all_probabilities = []
                for length, ngrams in negative_grams_by_length.items():
                    all_negative_grams.extend(ngrams)
                    if length in length_ratios:
                        all_probabilities.extend([length_ratios[length]] * len(ngrams))
                    else:
                        all_probabilities.extend([remaining_ratio / len(negative_grams_by_length)] * len(ngrams))

                # Normalize the probabilities
                all_probabilities = np.array(all_probabilities)
                all_probabilities /= all_probabilities.sum()

                # Weighted random choice function
                def weighted_random_choice(items, weights, k):
                    return np.random.choice(items, size=k, replace=False, p=weights)

                # Calculate the number of samples
                negative_example = math.ceil(positive_example * 1.5)

                # Sample negative_grams based on the probability distribution if the list is not empty
                if all_negative_grams:
                    k = min(negative_example, len(all_negative_grams))
                    selected_negative_grams = weighted_random_choice(all_negative_grams, all_probabilities, k).tolist()

                    for g in selected_negative_grams:
                        answer_sentence = '%s 不是一个命名实体' % g
                        csv_writer.writerow([first, answer_sentence])

def json2template_german(input_path, output_path, dataset='multiconer', generate_negative=False):
    """
    Convert JSON format data to CSV format data with template.
    Args:
        input_path: the path to the input file.
        output_path: the path to the output file.
        dataset: the dataset to generate template.
        generate_negative: a boolean variable to specify whether to generate negative template.
    """
    output_filename = output_path

    if dataset == 'multiconer':
        extend_label = {
            'LOC': 'Ort',  # Place/Location
            'PER': 'Person',  # Person
            'CORP': 'Unternehmen',  # Corporation
            'GRP': 'Gruppe',  # Group
            'PROD': 'Produkt',  # Product
            'CW': 'kreatives Werk'  # Creative Work
        }

    with open(input_path, 'r',  encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_filename, "w", newline='',  encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        # Write header row
        csv_writer.writerow(["Source sentence", "Answer sentence"])

        for item in data:
            tokens = item['text']
            labels = item['label']

            buffer_token = ""
            buffer_label = ""
            positive_example = 0
            first = " ".join(tokens)
            all_ngrams = generate_all_ngrams(first)
            positive_grams = []

            for l, t in zip(labels, tokens):
                if l.split("-")[0] != 'I' and buffer_token != "":
                    answer_sentence = '%s ist eine %s' % (
                        buffer_token, buffer_label) if buffer_label=='Person' or buffer_label== 'Gruppe' else '%s ist ein %s' % (
                        buffer_token, buffer_label)
                    positive_example += 1
                    positive_grams += generate_all_ngrams(buffer_token)
                    csv_writer.writerow([first, answer_sentence])
                    buffer_token = ""
                    buffer_label = ""
                if l.split("-")[0] == 'B':
                    buffer_token = t
                    buffer_label = extend_label[l.split("-")[1]]
                if l.split("-")[0] == 'I':
                    buffer_token += " " + t
            if buffer_token != "":
                answer_sentence = '%s ist eine %s' % (
                    buffer_token,
                    buffer_label) if buffer_label == 'Person' or buffer_label == 'Gruppe' else '%s ist ein %s' % (
                    buffer_token, buffer_label)
                csv_writer.writerow([first, answer_sentence])
                positive_example += 1
                positive_grams += generate_all_ngrams(buffer_token)

            if generate_negative:
                negative_grams = [item for item in all_ngrams if item not in positive_grams]

                # Separate negative_grams by length
                negative_grams_by_length = {}
                for item in negative_grams:
                    length = len(item.split())
                    if length not in negative_grams_by_length:
                        negative_grams_by_length[length] = []
                    negative_grams_by_length[length].append(item)

                # Define the ratio for each length, including lengths greater than 4
                length_ratios = {
                    1: 0.648,
                    2: 0.30,
                    3: 0.037,
                    4: 0.011,
                    5: 0.003
                }
                # Assign the remaining probability (0.02) to lengths greater than 4
                remaining_ratio = 0.001

                # Create a list of all negative_grams and their probabilities
                all_negative_grams = []
                all_probabilities = []
                for length, ngrams in negative_grams_by_length.items():
                    all_negative_grams.extend(ngrams)
                    if length in length_ratios:
                        all_probabilities.extend([length_ratios[length]] * len(ngrams))
                    else:
                        all_probabilities.extend([remaining_ratio / len(negative_grams_by_length)] * len(ngrams))

                # Normalize the probabilities
                all_probabilities = np.array(all_probabilities)
                all_probabilities /= all_probabilities.sum()

                # Weighted random choice function
                def weighted_random_choice(items, weights, k):
                    return np.random.choice(items, size=k, replace=False, p=weights)

                # Calculate the number of samples
                negative_example = math.ceil(positive_example * 1.5)

                # Sample negative_grams based on the probability distribution if the list is not empty
                if all_negative_grams:
                    k = min(negative_example, len(all_negative_grams))
                    selected_negative_grams = weighted_random_choice(all_negative_grams, all_probabilities, k).tolist()

                    for g in selected_negative_grams:
                        answer_sentence = '%s ist kein benanntes Entität' % g
                        csv_writer.writerow([first, answer_sentence])

def json2template_spanish(input_path, output_path, dataset='multiconer', generate_negative=False):
    """
    Convert JSON format data to CSV format data with template.
    Args:
        input_path: the path to the input file.
        output_path: the path to the output file.
        dataset: the dataset to generate template.
        generate_negative: a boolean variable to specify whether to generate negative template.
    """
    output_filename = output_path

    if dataset == 'multiconer':
        extend_label = {
            'LOC': 'lugar',  # Place/Location
            'PER': 'persona',  # Person
            'CORP': 'corporación',  # Corporation
            'GRP': 'grupo',  # Group
            'PROD': 'producto',  # Product
            'CW': 'obra creativa'  # Creative Work
        }

    with open(input_path, 'r',  encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_filename, "w", newline='',  encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        # Write header row
        csv_writer.writerow(["Source sentence", "Answer sentence"])

        for item in data:
            tokens = item['text']
            labels = item['label']

            buffer_token = ""
            buffer_label = ""
            positive_example = 0
            first = " ".join(tokens)
            all_ngrams = generate_all_ngrams(first)
            positive_grams = []

            for l, t in zip(labels, tokens):
                if l.split("-")[0] != 'I' and buffer_token != "":
                    answer_sentence = '%s es una entidad de %s' % (buffer_token, buffer_label)
                    positive_example += 1
                    positive_grams += generate_all_ngrams(buffer_token)
                    csv_writer.writerow([first, answer_sentence])
                    buffer_token = ""
                    buffer_label = ""
                if l.split("-")[0] == 'B':
                    buffer_token = t
                    buffer_label = extend_label[l.split("-")[1]]
                if l.split("-")[0] == 'I':
                    buffer_token += " " + t
            if buffer_token != "":
                answer_sentence = '%s es una entidad de %s' % (buffer_token, buffer_label)
                csv_writer.writerow([first, answer_sentence])
                positive_example += 1
                positive_grams += generate_all_ngrams(buffer_token)

            if generate_negative:
                negative_grams = [item for item in all_ngrams if item not in positive_grams]

                # Separate negative_grams by length
                negative_grams_by_length = {}
                for item in negative_grams:
                    length = len(item.split())
                    if length not in negative_grams_by_length:
                        negative_grams_by_length[length] = []
                    negative_grams_by_length[length].append(item)

                # Define the ratio for each length, including lengths greater than 4
                length_ratios = {
                    1: 0.648,
                    2: 0.30,
                    3: 0.037,
                    4: 0.011,
                    5: 0.003
                }
                # Assign the remaining probability (0.02) to lengths greater than 4
                remaining_ratio = 0.001

                # Create a list of all negative_grams and their probabilities
                all_negative_grams = []
                all_probabilities = []
                for length, ngrams in negative_grams_by_length.items():
                    all_negative_grams.extend(ngrams)
                    if length in length_ratios:
                        all_probabilities.extend([length_ratios[length]] * len(ngrams))
                    else:
                        all_probabilities.extend([remaining_ratio / len(negative_grams_by_length)] * len(ngrams))

                # Normalize the probabilities
                all_probabilities = np.array(all_probabilities)
                all_probabilities /= all_probabilities.sum()

                # Weighted random choice function
                def weighted_random_choice(items, weights, k):
                    return np.random.choice(items, size=k, replace=False, p=weights)

                # Calculate the number of samples
                negative_example = math.ceil(positive_example * 1.5)

                # Sample negative_grams based on the probability distribution if the list is not empty
                if all_negative_grams:
                    k = min(negative_example, len(all_negative_grams))
                    selected_negative_grams = weighted_random_choice(all_negative_grams, all_probabilities, k).tolist()

                    for g in selected_negative_grams:
                        answer_sentence = '%s no es una entidad nombrada' % g
                        csv_writer.writerow([first, answer_sentence])
def conll_to_txt(conll_file_path, output_file_path):
    with open(conll_file_path, "r", encoding="utf-8") as f:
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line.strip() == "":
                    if words:
                        for word, label in zip(words, labels):
                            out_f.write(f"{word} {label}\n")
                        out_f.write("\n")
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].strip())
                    else:
                        labels.append("O")
            if words:
                for word, label in zip(words, labels):
                    out_f.write(f"{word} {label}\n")
                out_f.write("\n")

if __name__ == '__main__':
    language = ['FA-Farsi', 'HI-Hindi','KO-Korean','NL-Dutch', 'RU-Russian', 'TR-Turkish', 'ZH-Chinese']
    for lan in language:

        short = lan.split('-')[0].lower()
        json2template(f'./data/{lan}/5shots.json', f'./data/{lan}/5shots.csv', generate_negative=True)
        json2template(f'./data/{lan}/10shots.json', f'./data/{lan}/10shots.csv', generate_negative=True)
        json2template(f'./data/{lan}/20shots.json', f'./data/{lan}/20shots.csv', generate_negative=True)
        json2template(f'./data/{lan}/50shots.json', f'./data/{lan}/50shots.csv', generate_negative=True)
        json2template(f'./data/{lan}/dev.json', f'./data/{lan}/dev.csv', generate_negative=False)
        conll_to_txt(f'./data/{lan}/{short}_test.conll', f'./data/{lan}/test.txt')

    # conll_to_txt('./data/BN-Bangla/bn_test.conll', './data/BN-Bangla/test.txt')