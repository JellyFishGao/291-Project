from utils_metrics import get_entities_bio, f1_score, precision_score, recall_score, classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, MBartForConditionalGeneration, MBart50Tokenizer
#from sequence_labeling_model import SequenceLabelingModel

import torch
import time
import math
import argparse
import random

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def template_multiconer_entity(words, input_TXT, start, tokenizer, model, device):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    bangla_template_list = [
        " একটি ব্যক্তি সত্তা .",
        " একটি স্থান সত্তা .",
        " একটি কর্পোরেশন সত্তা .",
        " একটি দল সত্তা .",
        " একটি পণ্য সত্তা .",
        " একটি সৃজনশীল কাজ সত্তা .",
        " একটি নামকরণ করা সত্তা নয় ."
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(bangla_template_list)):
            temp_list.append(words[i] + bangla_template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index,end_index,label,score]

def template_multiconer_entity_en(words, input_TXT, start, tokenizer, model, device):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [" is a person entity .", " is a location entity .", " is a corporation entity .",
                     " is a group entity .", " is a product entity .", " is a creative work entity .",
                     " is not a named entity ."]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index,end_index,label,score]

def template_multiconer_entity_zh(words, input_TXT, start, tokenizer, model, device):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " 是一个人名实体。",  # Person entity
        " 是一个地点实体。",  # Location entity
        " 是一个公司实体。",  # Corporation entity
        " 是一个群体实体。",  # Group entity
        " 是一个产品实体。",  # Product entity
        " 是一个创作作品实体。",  # Creative work entity
        " 不是一个命名实体。"  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    # score_list.append(score)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index,end_index,label,score]

def template_multiconer_entity_de(words, input_TXT, start, tokenizer, model, device):
    # Eingabetext -> Vorlage
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " ist eine Person.",  # Person entity
        " ist ein Ort.",  # Location entity
        " ist ein Unternehmen.",  # Corporation entity
        " ist eine Gruppe.",  # Group entity
        " ist ein Produkt.",  # Product entity
        " ist ein kreatives Werk.",  # Creative work entity
        " ist kein benanntes Entität."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_es(words, input_TXT, start, tokenizer, model, device):
    # texto de entrada -> plantilla
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " es una entidad de persona.",  # Person entity
        " es una entidad de lugar.",  # Location entity
        " es una entidad de corporación.",  # Corporation entity
        " es una entidad de grupo.",  # Group entity
        " es una entidad de producto.",  # Product entity
        " es una entidad de obra creativa.",  # Creative work entity
        " no es una entidad nombrada."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_fa(words, input_TXT, start, tokenizer, model, device):
    # متن ورودی -> قالب
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " یک موجودیت شخص است.",  # Person entity
        " یک موجودیت مکان است.",  # Location entity
        " یک موجودیت شرکت است.",  # Corporation entity
        " یک موجودیت گروه است.",  # Group entity
        " یک موجودیت محصول است.",  # Product entity
        " یک موجودیت اثر خلاقانه است.",  # Creative work entity
        " یک موجودیت نام‌گذاری‌شده نیست."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_hi(words, input_TXT, start, tokenizer, model, device):
    # इनपुट टेक्स्ट -> टेम्पलेट
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " एक व्यक्ति इकाई है।",  # Person entity
        " एक स्थान इकाई है।",  # Location entity
        " एक कंपनी इकाई है।",  # Corporation entity
        " एक समूह इकाई है।",  # Group entity
        " एक उत्पाद इकाई है।",  # Product entity
        " एक रचनात्मक कार्य इकाई है।",  # Creative work entity
        " एक नामित इकाई नहीं है।"  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_ko(words, input_TXT, start, tokenizer, model, device):
    # 입력 텍스트 -> 템플릿
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " 는 사람 명사입니다.",  # Person entity
        " 는 장소 명사입니다.",  # Location entity
        " 는 회사 명사입니다.",  # Corporation entity
        " 는 그룹 명사입니다.",  # Group entity
        " 는 제품 명사입니다.",  # Product entity
        " 는 창작물 명사입니다.",  # Creative work entity
        " 는 명명된 명사가 아닙니다."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_nl(words, input_TXT, start, tokenizer, model, device):
    # invoertekst -> sjabloon
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " is een persoonsentiteit.",  # Person entity
        " is een plaatsentiteit.",  # Location entity
        " is een bedrijfseenheid.",  # Corporation entity
        " is een groepseenheid.",  # Group entity
        " is een producteenheid.",  # Product entity
        " is een creatieve werkentiteit.",  # Creative work entity
        " is geen benoemde entiteit."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_ru(words, input_TXT, start, tokenizer, model, device):
    # входной текст -> шаблон
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " это сущность человека.",  # Person entity
        " это сущность места.",  # Location entity
        " это сущность компании.",  # Corporation entity
        " это сущность группы.",  # Group entity
        " это сущность продукта.",  # Product entity
        " это сущность творческого произведения.",  # Creative work entity
        " это неименованная сущность."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def template_multiconer_entity_tr(words, input_TXT, start, tokenizer, model, device):
    # giriş metni -> şablon
    words_length = len(words)
    words_length_list = [len(i) for i in words]
    input_TXT = [input_TXT] * (7 * words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = [
        " bir kişi varlığıdır.",  # Person entity
        " bir yer varlığıdır.",  # Location entity
        " bir şirket varlığıdır.",  # Corporation entity
        " bir grup varlığıdır.",  # Group entity
        " bir ürün varlığıdır.",  # Product entity
        " bir yaratıcı eser varlığıdır.",  # Creative work entity
        " bir adlandırılmış varlık değildir."  # Not a named entity
    ]
    entity_dict = {0: 'PER', 1: 'LOC', 2: 'CORP', 3: 'GRP', 4: 'PROD', 5: 'CW', 6: 'O'}
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    output_length_list = [0] * 7 * words_length

    for i in range(len(temp_list) // 7):
        base_length = \
            ((tokenizer(temp_list[i * 7], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[
                1] - 4
        output_length_list[i * 7:i * 7 + 7] = [base_length] * 7
        output_length_list[i * 7 + 6] += 1

    score = [1] * 7 * words_length
    with torch.no_grad():
        output = \
            model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[
                0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(0, 7 * words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start + (score.index(max(score)) // 7)
    return [start, end, entity_dict[(score.index(max(score)) % 7)], max(score)]  # [start_index, end_index, label, score]

def prediction(input_TXT, tokenizer, model, device, dataset_name):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i + j])
            words.append(word)

        if dataset_name == 'zh':
            entity = template_multiconer_entity_zh(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'de':
            entity = template_multiconer_entity_de(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'es':
            entity = template_multiconer_entity_es(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'fa':
            entity = template_multiconer_entity_fa(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'hi':
            entity = template_multiconer_entity_hi(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'ko':
            entity = template_multiconer_entity_ko(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'nl':
            entity = template_multiconer_entity_nl(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'ru':
            entity = template_multiconer_entity_ru(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'tr':
            entity = template_multiconer_entity_tr(words, input_TXT, i, tokenizer, model, device)
        elif dataset_name == 'en':
            entity = template_multiconer_entity_en(words, input_TXT, i, tokenizer, model, device)


        if entity[1] >= len(input_TXT_list):
            entity[1] = len(input_TXT_list) - 1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i + 1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)

    for entity in entity_list:
        label_list[entity[0]:entity[1] + 1] = ["I-" + entity[2]] * (entity[1] - entity[0] + 1)
        label_list[entity[0]] = "B-" + entity[2]
    return label_list


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def template_based_inference(model_path='facebook/mbart-large-50', input_test_data="./source/conll2003/test.txt",
                             dataset_name="conll2003"):
    tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
    # input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    model.config.use_cache = False
    # input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    # print(input_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    score_list = []
    file_path = input_test_data
    guid_index = 1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("#") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))

    # Randomly select 5000 examples
    random.seed(42)  # For reproducibility
    if len(examples) > 5000:
        examples = random.sample(examples, 300)
    trues_list = []
    preds_list = []
    str = ' '
    num_01 = len(examples)
    num_point = 0
    start = time.time()
    for example in examples:
        sources = str.join(example.words)
        preds_list.append(prediction(sources, tokenizer, model, device, dataset_name))
        trues_list.append(example.labels)
        print('%d/%d (%s)' % (num_point + 1, num_01, cal_time(start)))
        print('Pred:', preds_list[num_point])
        print('Gold:', trues_list[num_point])
        num_point += 1

    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    results = {
        "precision": precision_score(true_entities, pred_entities),
        "recall": recall_score(true_entities, pred_entities),
        "f1": f1_score(true_entities, pred_entities),
    }
    print('Precision: ', results["precision"], '\nRecall: ', results["recall"], '\nF1: ', results["f1"])
    report = classification_report(true_entities, pred_entities)
    print(report)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Inference pipeline for a template-based BART model.')

    # Add the model path argument
    parser.add_argument('--model_path', type=str, default='./outputs/best_model', help='path to the model')

    # Add the input test data argument
    parser.add_argument('--input_test_data', type=str, default='./data/BN-Bangla/test.txt',
                        help='path to input testing data')

    # Add the NER method argument
    parser.add_argument('--method_name', type=str, default='template_based',
                        help='method used for NER')

    # Add the evaluation dataset argument
    parser.add_argument('--dataset_name', type=str, default='en', help='path to evaluation dataset')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the inference function with the command line arguments as input
    if args.method_name == 'template_based':
        template_based_inference(args.model_path, args.input_test_data, args.dataset_name)
    # elif args.method_name == 'sequence_labeling':
    #     sequence_labeling_inference(args.model_path, args.input_test_data, args.dataset_name)
    else:
        print("please enter the correct method name")