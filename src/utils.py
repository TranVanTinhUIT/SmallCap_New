from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):

    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids
    else:  
        return input_ids, label_ids

def prep_strings_append_score(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    # retrieved_caps: [{'caption': 'xxxx', distance: xxx}]
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = build_infix_from_retrieved_caps(retrieved_caps, k)
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids, prefix_ids
    else:  
        return input_ids, label_ids, prefix_ids

def build_infix_from_retrieved_caps(retrieved_caps, k):
  caps = sorted(retrieved_caps, key=lambda x: x["distance"], reverse=True)[:k]
  prompt = ', \n\n'.join(['"{}" with cosin similariry is {}'.format(cap['caption'], cap['distance']) for cap in caps]) + '.'
  return prompt

def build_infix_list_prompt_from_retrieved_caps(retrieved_caps, k):
  # output prompt format: 
  #         Similar images and cosine scores following:
  #          - caption1: 0.25
  #          - caption2: 0.26
  #          - caption3: 0.19
  #          - caption4: 0.0.1
  #         ...
  caps = sorted(retrieved_caps, key=lambda x: x["distance"], reverse=True)[:k]
  prompt = '\r\n'.join(['- {}: {}'.format(cap['caption'], cap['distance']) for cap in caps]) + '.'
  return prompt

def prep_strings_stylish(text, tokenizer, styles, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    # styles => see definition in stylish_Definition.json (length, emotion)
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True 
        truncation = True
    
    if retrieved_caps is not None:
        infix = build_infix_from_retrieved_caps(retrieved_caps, k)
        prefix = template.replace('||', infix)
        prefix = prefix.format(styles) # Replace styles
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id] 
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    
    if is_test:
        return input_ids, prefix_ids
    else:  
        return input_ids, label_ids, prefix_ids

def build_styles(tokens, styleDefinition): 
    styles = {}
    # length
    if len(tokens) <= styleDefinition['length']['Short']['max']:
        styles['length'] = 'short'
    elif len(tokens) > styleDefinition['length']['Long']['min']:
        styles['length'] = 'long'
    else: 
        styles['length'] = 'medium'
    
    # emotion
    trimmed_tokens = [item.strip() for item in tokens]
    trimmed_positive = [item.strip() for item in styleDefinition['emotion']['positive']]
    trimmed_negative = [item.strip() for item in styleDefinition['emotion']['negative']]

    common_positive = set(trimmed_tokens) & set(trimmed_positive)
    common_negative = set(trimmed_tokens) & set(trimmed_negative)

    if len(common_positive) == len(common_negative):
        styles['emotion'] = 'normal'
    elif len(common_positive) > len(common_negative):
        styles['emotion'] = 'positive'
    else:
        styles['emotion'] = 'negative'

    return styles


def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.startswith(tokenizer.bos_token):
        pred = pred[len(tokenizer.bos_token):]
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-len(tokenizer.eos_token)]
    return pred

class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_caption_length=25):
        self.df = df
        self.tokenizer = tokenizer
        self.features = h5py.File(features_path, 'r')
        self.max_target_length = max_caption_length

        if rag:
            self.template = open(template_path).read().strip() + ' '
            self.max_target_length = (max_caption_length  # target caption
                                     + max_caption_length * k # retrieved captions
                                     + len(tokenizer.encode(self.template)) # template
                                     + len(tokenizer.encode('\n\n')) * (k-1) # separator between captions
                                     )
            assert k is not None 
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        styles = self.df['styles'][idx]
        if self.rag: 
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings_stylish(text, self.tokenizer, styles, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings_stylish(text, self.tokenizer, styles, max_length=self.max_target_length)
        # load precomputed features
        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs), 
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 

def load_data_for_training_with_styles(annot_path, style_def_path, caps_path=None):
    style_definition = json.load(open(style_def_path))
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens']), 'styles': build_styles(sentence['tokens'], style_definition)})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data 

def load_data_for_inference(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'test': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
        if item['split'] == 'test':
            data['test'].append(image)
        elif item['split'] == 'val':
            data['val'].append(image)

    return data      

