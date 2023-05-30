#!/usr/bin/env python3


import collections
import json
import argparse
import itertools
from typing import Optional

import numpy as np

import util


PEANO_MODEL_PREFIX = 'peano-'
STAR_PROBLEMS_PER_ITERATION = 40
STAR_LAST_FULL_ITERATION = 2


def format_reasoner_name(raw_name: str) -> str:
    if raw_name.startswith(PEANO_MODEL_PREFIX):
        name = raw_name[len(PEANO_MODEL_PREFIX):] + ' + Guide'
        return name

    return raw_name


def syllogism_dataset_name(raw_name: str):
    if 'nonsense' in raw_name:
        return 'Nonsense'
    elif 'inconsistent' in raw_name:
        return 'Inconsistent'

    assert 'consistent' in raw_name
    return 'Consistent'


def format_base_reasoner_name(raw_name: str) -> str:
    if raw_name.endswith('text-davinci-003'):
        return 'GPT-3 Davinci'
    if raw_name.endswith('gpt-3.5-turbo'):
        return 'GPT-3.5 Turbo'
    if raw_name.endswith('llama-13b-hf'):
        return 'LLaMA 13B'

    raise ValueError(f'Unknown reasoner {raw_name}')


def format_star_mode_name(raw_name: str):
    if 'direct' in raw_name:
        return 'Unguided'
    if 'selective' in raw_name:
        return 'Strict Guided'
    return 'Guided'


def has_guide(raw_name: str) -> bool:
    if raw_name.startswith('peano'):
        return True
    if 'llama' in raw_name:
        return 'direct' not in raw_name
    return False


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


def load_results(paths: list[str], dataset_filter: Optional[str],
                 unknown_as_random_guess=True) -> list:
    records = []

    for path in paths:
        with open(path) as f:
            results_obj = json.load(f)

        print('Loaded', path)
        records.extend(results_obj.values())

    for r in records:
        if not r['reasoning']:
            r['correct'] = 0.5
            continue

        if r['prediction'] in ('No', 'False') or 'Answer: No' in r['reasoning']:
            r['prediction'] = False
        if r['prediction'] in ('Yes', 'True') or 'Answer: Yes' in r['reasoning']:
            r['prediction'] = True

        if r['prediction'] not in (True, False):
            print(r['prediction'])
            r['correct'] = 0.5
            continue

        r['correct'] = r['prediction'] == r['answer']

        #if unknown_as_random_guess and (r['prediction'] not in ('True', 'False')):
        #    r['correct'] = 0.5

    return [r
            for r in records
            if (dataset_filter or '') in r['dataset']]


def compute_success_rates(records: list, strict=False) -> dict:
    successes = collections.defaultdict(list)
    result = {}

    for r in records:
        correct = float(r['correct'])

        #if not strict and (r['prediction'] not in ('True', 'False')):
        #    correct = 0.5

        successes[(r['dataset'], r['reasoner'])].append(correct)



    for k, v in successes.items():
        result[k] = np.mean(v)

    return result


def report_model(name):
    return 'davinci' in name or 'gpt-3.5' in name or 'llama-13b' in name


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


def generate_multihop_reasoning_plot(results_path: str, output_path: str):
    results = load_results(results_path, None, True)
    data = []

    for r in results:
        if not report_model(r['reasoner']):
            continue

        if 'syllogism' in r['dataset']:
            continue

        # LLaMA results were first 40 problems on 3 different seeds
        # rather than problems up to 120 in same seed.
        if 'llama' in r['reasoner'] and ('proofwriter' not in r['dataset']) \
           and int(r['problem'][len('example'):]) >= 40:
            continue

        r['model'] = format_reasoner_name(r['reasoner'])
        r['base_model'] = format_base_reasoner_name(r['reasoner'])
        r['hops'] = get_dataset_name(r['dataset'])[2]
        r['dataset'] = base_dataset_name(r['dataset'])
        r['guide'] = 'Yes' if has_guide(r['reasoner']) else 'No'
        data.append(r)

    util.plot_vegalite('multihop-reasoning', data, output_path)


def generate_syllogism_plot(results_path: str, output_path: str):
    results = load_results(results_path, None)
    data = []

    for r in results:
        if not report_model(r['reasoner']):
            continue

        r['model'] = format_reasoner_name(r['reasoner'])
        r['base_model'] = format_base_reasoner_name(r['reasoner'])
        r['dataset'] = syllogism_dataset_name(r['dataset'])
        r['guide'] = 'Yes' if has_guide(r['reasoner']) else 'No'
        data.append(r)

    util.plot_vegalite('syllogism-validity', data, output_path)


def generate_star_plot(results_path: str, output_path: str):
    results = load_results(results_path, None)
    data = []

    for r in results:
        if not report_model(r['reasoner']):
            continue

        r['model'] = format_reasoner_name(r['reasoner'])
        r['iteration'] = int(r['problem'][len('example'):]) // STAR_PROBLEMS_PER_ITERATION

        if r['iteration'] > STAR_LAST_FULL_ITERATION:
            continue

        r['mode'] = format_star_mode_name(r['reasoner'])
        data.append(r)

    util.plot_vegalite('star', data, output_path)


def compute_dataset_stats():
    import lm_tool

    prontoqa = lm_tool.PrOntoQADataset('prontoqa', [])
    proofwriter = lm_tool.PrOntoQADataset('proofwriter', [])

    for hops in [1]:
        for ontology in ['', '_trueontology', '_falseontology']:
            path = f'./prontoqa/{hops}hop_random{ontology}_seed19.json'
            d = lm_tool.PrOntoQADataset.load(path)
            prontoqa.problems.extend(d.problems)

        path = f'./proofwriter/proofwriter_{hops}hop.json'
        d = lm_tool.PrOntoQADataset.load(path)
        proofwriter.problems.extend(d.problems)

    def median_assumptions(ds):
        return np.median([len(p.test_example.theory) for p in ds.problems])

    print('Median context size for PrOntoQA:', median_assumptions(prontoqa))
    print('Median context size for ProofWriter:', median_assumptions(proofwriter))


def compute_deontic_results():
    results = load_results(['deontic-results.json'], 'deontic')

    srs = compute_success_rates(results)

    print(srs)


def compute_solutions_stats():
    import lm_tool

    results = json.load(open('results-0514.json'))

    guided_results = [r for r in results.values()
                      if 'peano' in r['reasoner'] and report_model(r['reasoner'])]

    unguided_results = [r for r in results.values()
                        if 'peano' not in r['reasoner'] and report_model(r['reasoner'])]

    guided_failures = [r for r in guided_results if not r['correct']]
    certified_failures = [r for r in guided_results
                          if (r['prediction'] in ('True', 'False') and
                              r['answer'] != r['prediction'])]

    print('Fraction of results where model proved the goal and was wrong:', len(certified_failures) / len(guided_failures))

    exhausted_inferences = [r for r in guided_results
                            if r['reasoning'] and 'infer:nothing' in r['reasoning']]

    exhausted_and_unknown = [r for r in exhausted_inferences
                             if not ('Answer: True' in r['reasoning'] or 'Answer: False' in r['reasoning'])]

    exhausted_and_false = [r for r in exhausted_inferences
                             if 'Answer: False' in r['reasoning']]

    print('Fraction of exhausted reasoning where model abstains:', len(exhausted_and_unknown) / len(exhausted_inferences))
    print('Fraction of exhausted reasoning where model answers false:', len(exhausted_and_false) / len(exhausted_inferences))

    unguided_unknown = [r for r in unguided_results
                        if 'Answer:' in r['reasoning'] and '3.5' not in r['reasoner']
                        and not ('Answer: True' in r['reasoning'] or 'Answer: False' in r['reasoning'])]

    print(len(unguided_unknown), 'unknown results in not GPT-3.5 Turbo models.')

    unguided_unknown_gpt35 = [r for r in unguided_results
                        if 'Answer:' in r['reasoning'] and '3.5' in r['reasoner']
                        and not ('Answer: True' in r['reasoning'] or 'Answer: False' in r['reasoning'])]

    unguided_gpt35 = [r for r in unguided_results if '3.5' in r['reasoner']]

    print(len(unguided_unknown_gpt35) / len(unguided_gpt35), 'unknown results from GPT-3.5 Turbo')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', action='store_true',
                        help='Generate main table of results.')
    parser.add_argument('--merge', action='store_true',
                        help='Merge a list of results files into a single one')
    parser.add_argument('--plot-multihop-reasoning', action='store_true',
                        help='Generate line plot of multihop reasoning results.')
    parser.add_argument('--plot-syllogism', action='store_true',
                        help='Generate bar plot of syllogistic reasoning results.')
    parser.add_argument('--plot-star', action='store_true',
                        help='Generate line plot of STaR results.')
    parser.add_argument('--deontic-results', action='store_true',
                        help='Compute deontic logic results.')
    parser.add_argument('--stats', action='store_true',
                        help='Compute dataset stats.')
    parser.add_argument('--solution-stats', action='store_true',
                        help='Compute model solution stats.')
    parser.add_argument('--results', help='Path to results file.', nargs='*')
    parser.add_argument('--dataset-filter', default='',
                        help='Only consider datasets with this substring.')
    parser.add_argument('--output', default='/dev/null',
                        help='Path to output file.')

    opt = parser.parse_args()

    print(opt.results)

    if opt.table:
        generate_results_table(opt.results, opt.dataset_filter,
                               opt.output)
    elif opt.plot_multihop_reasoning:
        generate_multihop_reasoning_plot(opt.results, opt.output)
    elif opt.plot_syllogism:
        generate_syllogism_plot(opt.results, opt.output)
    elif opt.plot_star:
        generate_star_plot(opt.results, opt.output)
    elif opt.stats:
        compute_dataset_stats()
    elif opt.solution_stats:
        compute_solutions_stats()
    elif opt.deontic_results:
        compute_deontic_results()


if __name__ == '__main__':
    main()
