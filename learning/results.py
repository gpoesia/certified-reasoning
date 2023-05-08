#!/usr/bin/env python3


import collections
import json
import argparse
import itertools

import numpy as np


PEANO_MODEL_PREFIX = 'peano-'


def format_reasoner_name(raw_name: str) -> str:
    if raw_name.startswith(PEANO_MODEL_PREFIX):
        name = raw_name[len(PEANO_MODEL_PREFIX):] + ' + Guide'
        return name

    return raw_name


def get_dataset_name(raw_name: str) -> (str, str, str):
    if 'prontoqa' in raw_name:
        hops = None

        for i in itertools.count():
            if f'{i}hop' in raw_name:
                hops = i
                break

        ontology = ('True' if 'trueontology' in raw_name
                    else 'False' if 'falseontology' in raw_name else
                    'Fictitional')
        return ('PrOntoQA', ontology, hops)
    if 'proofwriter' in raw_name:
        for i in itertools.count():
            if f'{i}hop' in raw_name:
                hops = i
                break

        return ('ProofWriter', '', hops)

    assert False, 'Unknown dataset ' + raw_name


def base_dataset_name(raw_name: str) -> str:
    ds, ontology, _ = get_dataset_name(raw_name)
    return f'{ds} {ontology}'

def dataset_hops(raw_name: str) -> str:
    _, _, hops = get_dataset_name(raw_name)
    return f'{hops}H'

def format_dataset_name(raw_name: str) -> str:
    ds, ontology, hops = get_dataset_name(raw_name)
    return f'{ds} {ontology} {hops}H'

def load_results(path: str, dataset_filter: str) -> list:
    with open(path) as f:
        results_obj = json.load(f)
    print('Loaded', path)

    records = results_obj.values()

    return [r
            for r in records
            if r['error'] is None and dataset_filter in r['dataset']]


def compute_success_rates(records: list) -> dict:
    successes = collections.defaultdict(list)
    result = {}

    for r in records:
        successes[(r['dataset'], r['reasoner'])].append(int(r['correct']))

    for k, v in successes.items():
        result[k] = np.mean(v)

    return result


def report_model(name):
    return 'davinci' in name or 'gpt-3.5' in name


def make_table(records: list) -> str:
    models, datasets = [], []

    success_rates = compute_success_rates(records)

    for k in success_rates:
        if report_model(k[1]):
            datasets.append(k[0])
            models.append(k[1])

    lines = []

    models = list(set(models))
    models.sort()
    datasets = list(set(datasets))
    datasets.sort()

    base_name = base_dataset_name(datasets[0])

    lines.append(fr'\begin{{tabular}}{{l{" c" * len(datasets)}}}')
    lines.append(r'\toprule')
    lines.append(f'& \\multicolumn{{{len(datasets)}}}{{c}}{{{base_name}}} \\\\')

    lines.append('& '.join(fr'\textbf{{{m}}}'
                           for m in ['Model'] + list(map(dataset_hops,
                                                         datasets))) + '\\\\')

    lines.append(r'\midrule')

    for m in models:
        l = [format_reasoner_name(m)]

        for d in datasets:
            sr = success_rates.get((d, m))
            if sr is None:
                l.append('-')
            else:
                l.append(f'{sr:.3f}')

        lines.append(' & '.join(l) + '\\\\')

    lines.append(r'\bottomrule')

    lines.append(r'\end{tabular}')

    return '\n'.join(lines)


def generate_results_table(results_path: str, dataset_filter: str, output_path: str):
    results = load_results(results_path, dataset_filter)

    with open(output_path, 'w') as f:
        f.write(make_table(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store_true',
                        help='Generate main table of results.')

    parser.add_argument('--results', help='Path to results file.')
    parser.add_argument('--dataset-filter', default='',
                        help='Only consider datasets with this substring.')
    parser.add_argument('--output-latex', default='results_table.tex',
                        help='Path to output file.')

    opt = parser.parse_args()

    if opt.table:
        generate_results_table(opt.results, opt.dataset_filter,
                               opt.output_latex)


if __name__ == '__main__':
    main()
