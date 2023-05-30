#!/usr/bin/env python3

import argparse
import collections
import math
import random
import os
import logging
import json
import itertools
import subprocess

import torch
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


PAD = 0
BOS = 1
EOS = 2
POSITIVE = ord(';')
NEGATIVE = ord('$')
EMPTY = '\x03'

def count_parameters(model):
    return sum(math.prod(p.shape) for p in model.parameters())


def format_parameter_count(model):
    n = count_parameters(model)

    if n < 1000:
        return str(n)
    if n < 10**6:
        return f'{n // 10**3}K'
    if n < 10**9:
        return f'{n / 10**6:.1f}M'

    return f'{n / 10**6:.1f}B'


def encode_batch(b: list[str], device: torch.device, bos=True, eos=True) -> torch.LongTensor:
    if not b:
        return torch.tensor([], dtype=torch.long, device=device)

    max_len = max(map(len, b))

    return torch.tensor([[BOS] * bos +
                         list(map(ord, o)) +
                         [EOS] * eos +
                         [PAD] * (max_len - len(o))
                         for o in b],
                        dtype=torch.long,
                        device=device)

def decode_batch(b: torch.LongTensor) -> list[str]:
    return [''.join(chr(c) for c in row if c > EOS) for row in b]


def sample_batch(examples: list[str], batch_size: int) -> list[str]:
    'Samples a batch of examples with a bounded total number of characters'
    batch = []
    max_size = 0

    while True:
        example = random.choice(examples)
        max_size = max(max_size, len(example))

        if max_size * (1 + len(batch)) > batch_size:
            break

        batch.append(example)

    return batch


def batch_strings(s: list[str], batch_size: int) -> list[str]:
    '''Batches a list of strings into small batches with a bounded
    total number of characters in each.'''

    batches = []

    stack = s[::-1]

    while stack:
        batch = []
        max_size = 0

        while stack:
            example = stack.pop()
            max_size = max(max_size, len(example))

            if batch and max_size * (1 + len(batch)) > batch_size:
                stack.append(example)
                break

            batch.append(example)
        batches.append(batch)

    return batches


def log(x):
    'Safe version of log'
    return math.log(1e-50 + max(0, x))

def softmax(logits: torch.Tensor, temperature = 1.0):
    s = logits.exp() / temperature
    return s / s.sum()


def pop_max(l: list, key) -> (object, list):
    if not l:
        return None, l

    i_max = max(range(len(l)), key=lambda i: key(l[i]))
    l[-1], l[i_max] = l[i_max], l[-1]
    return l[-1], l[:-1]


def shuffle_state(s: str) -> str:
    'Perform data augmentation for Peano states by shuffling e-classes and e-nodes'

    eclasses = s.split('; ')
    random.shuffle(eclasses)

    for i in range(len(eclasses)):
        enodes, dtype = eclasses[i].split(' : ')
        # Strip '{' and '}'
        enodes = enodes.lstrip('{').rstrip('}').split('=')
        random.shuffle(enodes)

        eclasses[i] = f'{{{"=".join(enodes)}}} : {dtype}'

    return '; '.join(eclasses)


def parse_sexp(s: str, ptr: int = 0) -> (object, int):
    while ptr < len(s) and s[ptr] == ' ':
        ptr += 1

    if s[ptr] == '(':
        # Read list
        ptr += 1 # Consume (
        l = []
        while s[ptr] != ')':
            elem, ptr = parse_sexp(s, ptr)
            l.append(elem)
        ptr += 1 # Consume )
        return l, ptr
    else:
        # Read atom
        before = ptr
        while ptr < len(s) and s[ptr] not in ' ()':
            ptr += 1
        return s[before:ptr], ptr


def randomize_atoms(sexp, criteria, sample, mapping):
    if isinstance(sexp, str):
        if sexp in mapping:
            return mapping[sexp]

        if criteria(sexp):
            v = str(sample())
            mapping[sexp] = v
            return v

        return sexp
    return [randomize_atoms(s, criteria, sample, mapping) for s in sexp]


def format_sexp(sexp, level=0, indent=0):
    if isinstance(sexp, str):
        return level * indent * ' ' + sexp
    sep = ' ' if not indent else '\n  '
    return ((level * indent * ' ') +
            '(' + sep.join(map(lambda e: format_sexp(e, level + 1, indent), sexp)) + ')')


def toggle_infix(sexp):
    if isinstance(sexp, str):
        return sexp
    children = list(map(toggle_infix, sexp))
    if len(children) == 3:
        return [children[1], children[0], children[2]]
    return children


def format_infix(sexp_str: str) -> str:
    sexp, _ = parse_sexp(sexp_str)
    return format_sexp(toggle_infix(sexp))


def randomly_mask_atoms(sexp, probability):
    if isinstance(sexp, str):
        if random.random() < probability:
            return '?'
        return sexp

    return list(map(lambda elem: randomly_mask_atoms(elem, probability), sexp))


def randomly_mask_goal_terms(goal: str, probability=0.1) -> str:
    'Perform data augmentation for Peano goals by masking some sub-terms'

    sexp, _ = parse_sexp(goal)
    sexp = randomly_mask_atoms(sexp, probability)
    return format_sexp(sexp)


def get_device(cfg):
    if cfg is None:
        return torch.device('cpu')

    if isinstance(cfg, int):
        return torch.device(cfg)

    if cfg.get('gpu') is not None:
        return torch.device(cfg.gpu)

    return torch.device('cpu')


def choose_from_list(prompt, l, to_str=str):
    print(prompt)
    for i, e in enumerate(l):
        print(f'{i:2d} - ', to_str(e))

    return l[int(input('> '))]


def setup_wandb(cfg: DictConfig):
    if cfg.job.get("wandb_project"):
        with open_dict(cfg.job):
            cfg.job.cwd = os.getcwd()
        wandb.init(
                project=cfg.job.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        for key in logging.Logger.manager.loggerDict.keys():
            if key.startswith('wandb'):
                logging.getLogger(key).setLevel(logging.WARNING)
    else:
        # Disable wandb (i.e., make log() a no-op).
        wandb.log = lambda *args, **kwargs: None


def count_inversions(l: list):
    '''Counts the number of inversions in a list.
    Complexity: O(nk), where n = len(l) and k = len(set(l)); thus,
    works well if k is small.
    '''
    counts = collections.defaultdict(int)
    inversions = 0

    for i, val in enumerate(l):
        for key, cnt in counts.items():
            if key > val:
                inversions += cnt
        counts[val] += 1

    return inversions


def plot_vegalite(template: str, data: list, output_path: str):
    with open(f'vega-lite/{template}.json') as f:
        spec = json.load(f)

    spec['data'] = {'values': data}

    json_path = output_path + '.vl.json'

    with open(json_path, 'w') as f:
        json.dump(spec, f)

    with open(output_path, 'wb') as f:
        subprocess.run(['vl2png', json_path, '--scale', '5'], stdout=f, check=True)


def bootstrap_mean_ci(trials, confidence):
    estimates = []

    for i in range(5000):
        estimates.append(np.mean(random.choices(trials, k=len(trials))))

    bounds = ((1 - confidence) / 2, 1 - (1 - confidence) / 2)
    return np.mean(estimates), tuple(np.quantile(estimates, bounds, method='nearest'))


# Simple script to convert the ProofWriter dataset into the PrOntoQA format.
def proofwriter_to_prontoqa(root):
    random.seed(19)

    for depth in [1, 2, 3, 5]:
        problems = []

        with open(os.path.join(root, f'depth-{depth}', 'meta-train.jsonl')) as f:
            records = list(map(json.loads, f.readlines()))

        for r in records:
            context = []

            for _, v in itertools.chain(r['triples'].items(), r['rules'].items()):
                context.append(v['text'])

            items = list(r['questions'].items())
            random.shuffle(items)

            for _, v in items:
                # Ignore 'Unknown' cases (no formal proof for those).
                if v['strategy'] not in ('proof', 'inv-proof'):
                    continue

                if v['QDep'] < max(1, depth - 1):
                    continue

                # Same format for questions as PrOntoQA.
                question = f'True or false: {v["question"]}'
                answer = str(v['answer'])
                chain_of_thought = []

                for key, step in v['proofsWithIntermediates'][-1]['intermediates'].items():
                    # Format of keys is 'intX' where X is an index; ignore the 'int'.
                    chain_of_thought.append((int(key[3:]), step['text']))

                chain_of_thought.sort(key=lambda kv: kv[0], reverse=True)
                chain_of_thought = [v for i, v in chain_of_thought]

                problems.append({
                    'question': ' '.join(context),
                    'query': question,
                    'chain_of_thought': chain_of_thought,
                    'answer': answer,
                })

                # Just take one problem from each theory.
                break

        random.shuffle(problems)

        N = len(problems)
        train, test = problems[:N//2], problems[N//2:]

        dataset = collections.OrderedDict()

        for i, p in enumerate(test):
            # PrOntoQA always has 4 in-context examples.
            incontext_examples = random.sample(train, k=4)

            p_obj = collections.OrderedDict()

            for j, e in enumerate(incontext_examples):
                p_obj[f'in_context_example{j}'] = e

            p_obj['test_example'] = p
            dataset[f'example{i+1}'] = p_obj

        output_path = f'proofwriter/proofwriter_{depth}hop.json'

        with open(output_path, 'w') as out:
            json.dump(dataset, out, indent=2)

        print('Wrote', output_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proofwriter-to-prontoqa', action='store_true',
                        help='Convert ProofWriter dataset to the PrOntoQA format.')
    parser.add_argument('--proofwriter-root', type=str,
                        help='Root of the ProofWriter dataset.')

    args = parser.parse_args()

    if args.proofwriter_to_prontoqa:
        proofwriter_to_prontoqa(args.proofwriter_root)

if __name__ == '__main__':
    main()
