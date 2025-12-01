import torch
import random
import numpy as np
import os
import logging
import sys

import spacy
from tqdm import tqdm
import open_clip
import h5py
import csv
import pandas as pd


def related_obj_extract(referit_data, object_file_path, clip_path, clip_threshold, maps, device):
    nlp = spacy.load('en_core_web_sm')
    clip, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=clip_path)
    clip = clip.to(device)
    pbar = tqdm(zip(referit_data['utterance'], referit_data['scan_id'], referit_data['stimulus_id']))
    pbar.set_description("Extract related objects")
    res_lists = []
    for text, scan_id, stimulus_id in pbar:
        doc = nlp(text)
        nouns = []
        for nc in doc.noun_chunks:
            x = nc.text.split(' ')
            if x[-1] in maps or (len(x) > 1 and x[-2] + ' ' + x[-1] in maps):
                nouns.append(nc.text)
        # nouns = ['the ' + nc.text for nc in doc if nc.pos_ == 'NOUN' and nc.text in maps]
        if len(nouns) == 0:
            res_lists.append([stimulus_id, '-1'])
            continue
        with torch.no_grad():
            tokenized = open_clip.tokenize(nouns)
            text_features = clip.encode_text(tokenized.to(device))
            with h5py.File(object_file_path, "r") as f:
                object_ids = np.array(f[scan_id]['object_ids'])
                object_features = np.array(f[scan_id]['object_features'])
            object_features = torch.from_numpy(object_features).to(device)
            object_features /= object_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            sim = object_features @ text_features.T
            text_probs, _ = torch.max(sim, dim=-1)
            num = torch.sum(text_probs > 0.2).int()
            inds = torch.argsort(text_probs, descending=True)
            related_object_ids = object_ids[inds.cpu().numpy()][:min(num, 20)]
            # text_probs, _ = torch.max(object_features @ text_features.T, dim=-1)
            # related_object_ids = object_ids[text_probs.cpu().numpy() > clip_threshold]
        if len(related_object_ids) == 0:
            res_lists.append([stimulus_id, '-1'])
        else:
            res_lists.append([stimulus_id, ','.join(related_object_ids.astype(str))])
    del clip
    df = pd.DataFrame(res_lists, columns=['stimulus_id', 'related_objects'])
    df.to_csv('./data/related_objects.csv', index=False)


def seed_training_code(manual_seed, strict=False):
    """Control pseudo-randomness for reproducibility.
    :param manual_seed: (int) random-seed
    :param strict: (boolean) if True, cudnn operates in a deterministic manner
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu_to_zero_position(real_gpu_loc):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(real_gpu_loc)


def create_logger(log_dir, std_out=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Add logging to file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add stdout to also print statements there
    if std_out:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
