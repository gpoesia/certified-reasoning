#!/usr/bin/env python3

'''
Fact retrieval (BeerQA dataset).
'''

import datetime
import json
import time
import random
import re
import openai
import domain

from dataclasses import dataclass
from typing import List, Dict, Optional

from completion_engine import CompletionEngine
from language_model import OpenAIModel, LanguageModel, download_or_use_cached, filter_maximal_tokens
from synchromesh import predict_constrained

from completion import PeanoCompletionEngine, ImplicitRetrievalCompletionEngine
from lm_tool import NaturalLanguageReasoner, OpenAIChatModel

# Matches the JSON structure of a BeerQA problem
@dataclass
class BeerQAExample:
    id: str
    src: str
    answers: list[str]
    question: str
    context: list[str]

# Convenience class created to represent different sets of BeerQA problems. 
# For example, 1 hop versus 2 hop problems
class BeerQAProblem:
    split: str
    questions: list[BeerQAExample]

@dataclass
class BeerQADataset:
    id: str      # this is just the path
    problems: list[BeerQAProblem]

    @staticmethod
    def load(path: str) -> 'BeerQADataset':
        with open(path, 'rb') as f:
            data = json.load(f)

        problems = []

        if "split" in data:
            dataset_split = data["split"]

        if "data" in data:
            for item in data["data"]:

                example = BeerQAExample(
                    id = item['id'],
                    src = item['src'],
                    answers = item['answers'],
                    question = item['question'],
                    context = item['context']
                )
                problems.append(example)
        
        return BeerQADataset(split=dataset_split, problems=problems)
    

# Matches the JSON structure of a BeerQA problem
@dataclass
class QasperQAExample:
    pass

# Convenience class created to represent different sets of BeerQA problems. 
# For example, 1 hop versus 2 hop problems
class QasperQAProblem:
    pass

@dataclass
class QasperQADataset:
    pass

class RateLimiter:
    def __init__(self, rpm: int = 20):
        self._min_interval = datetime.timedelta(seconds = 60 / rpm)
        self._last_request = datetime.datetime.now()

    def wait(self):
        'Wait enough to not hit the rate limit.'
        now = datetime.datetime.now()

        if self._last_request is not None:
            wait = (self._min_interval - (now - self._last_request)).total_seconds()
            if wait > 0:
                print('wait', wait)
            time.sleep(max(0, wait))

        self._last_request = datetime.datetime.now()

rate_limiter = RateLimiter(1000)

def _split_question(question: str):
    return [s + '.' for s in question.rstrip('.').split('. ')]

# TODO: implement a factual reasoner using PeanoLMReasoner as a reference
class FactualLMReasoner(NaturalLanguageReasoner):
    def __init__(self, completion_engine: CompletionEngine,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._separator = '###'

    def name(self) -> str:
        pass

    def prepare_for(self, dataset: str):
        pass

    def _format_problem(self, problem: object) -> str:
        pass

    def predict_answer(self, problem: BeerQAProblem) -> bool:
        pass

# TODO: implement a factual reasoner using PeanoChatLMReasoner as a reference
class FactualChatLMReasoner(NaturalLanguageReasoner):
    def __init__(self, completion_engine: CompletionEngine,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._separator = '###'

    def name(self) -> str:
        pass

    def prepare_for(self, dataset: str):
        pass

    def _format_problem(self, problem: object) -> str:
        pass

    def predict_answer(self, problem: BeerQAProblem) -> bool:
        pass

class PeanoChatLMReasoner(NaturalLanguageReasoner):
    def __init__(self, completion_engine: CompletionEngine,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._separator = '###'

    def name(self) -> str:
        return f'peano-{self._model}'

    def prepare_for(self, dataset: str):
        if 'trueontology' in dataset:
            prompt_file = 'prompts/peano_prontoqa_trueontology_short_prompt'
        elif 'falseontology' in dataset:
            prompt_file = 'prompts/peano_prontoqa_falseontology_short_prompt'
        elif 'proofwriter' in dataset:
            prompt_file = 'prompts/peano_proofwriter_short'
        elif 'realistic-consistent' in dataset:
            prompt_file = 'prompts/peano_syllogism_realistic_consistent_prompt'
        elif 'realistic-inconsistent' in dataset:
            prompt_file = 'prompts/peano_syllogism_realistic_inconsistent_prompt'
        elif 'syllogism-nonsense' in dataset:
            prompt_file = 'prompts/peano_syllogism_nonsense_prompt'
        else:
            prompt_file = 'prompts/peano_prontoqa_short_prompt'


        with open(prompt_file, 'r') as f:
            self._prompt = f.read().strip()
            print('Loaded prompt from', prompt_file)

    def _format_problem(self, problem: object) -> str:
        context = f' '.join(f'{i+1}- {sentence}'
                            for i, sentence in enumerate(problem.test_example.theory))
        query = problem.test_example.query
        return f'Context: {context}\nQuery: {query.strip()}'

    def predict_answer(self, problem: PrOntoQAProblem) -> bool:
        prompt = f'{self._prompt}\n{self._format_problem(problem)}'

        lm = OpenAIModel(self._model,
                         prompt,
                         temperature=self._temperature,
                         before_prediction_hook=rate_limiter.wait,
                         cache_path='.openai_cache'
                         )

        response = predict_constrained(self._completion_engine, lm, batch_size=800,
                                       stop_tokens=[self._separator])
        result = self._completion_engine.is_complete(response)

        done, answer = result if result is not None else (False, None)

        if not done:
            return 'Unknown', response

        return str(answer), response


class PeanoChatLMReasoner(NaturalLanguageReasoner):
    def __init__(self, completion_engine: CompletionEngine,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._index = 2

    def name(self) -> str:
        return f'peano-chat-{self._model}'

    def prepare_for(self, dataset: str):
        if 'trueontology' in dataset:
            prompt_file = 'prompts/peano_chat_prontoqa_trueontology_short_prompt'
        elif 'falseontology' in dataset:
            prompt_file = 'prompts/peano_chat_prontoqa_falseontology_short_prompt'
        elif 'proofwriter' in dataset:
            prompt_file = 'prompts/peano_chat_proofwriter'
        elif 'syllogism' in dataset:
            prompt_file = 'prompts/peano_chat_syllogism_nonsense_prompt'
            self._index = 3
        else:
            prompt_file = 'prompts/peano_chat_prontoqa_short_prompt'

        with open(prompt_file, 'r') as f:
            self._prompt = json.load(f)
            print('Loaded chat prompt from', prompt_file)

    def _format_problem(self, problem: object) -> List[Dict[str, str]]:
        context = f' '.join(f'{i+1}- {sentence}'
                            for i, sentence in enumerate(problem.test_example.theory))
        query = problem.test_example.query
        chat_problem = [{"role": "user", "content": f"Problem #{self._index}\nContext: {context}\nQuery: {query.strip()}"}]
        return chat_problem
    
    def predict_answer(self, problem: object) -> bool:

        test = self._format_problem(problem)
        prompt_messages = self._prompt + test

        lm = OpenAIChatModel(self._model,
                             prompt_messages,
                             temperature=self._temperature,
                             before_prediction_hook=rate_limiter.wait,
                             cache_path='.openai_cache')

        response = predict_constrained(self._completion_engine, lm, batch_size=800)
        done, answer = self._completion_engine.is_complete(response)

        if not done:
            return 'Unknown', response

        return str(answer), response

# TODO: refactor 
def evaluate_reasoner(results_path: str,
                      dataset: PrOntoQADataset,
                      reasoner: NaturalLanguageReasoner,
                      max_problems: int = None):
    success = []

    print('Evaluating', reasoner.name(), 'on', dataset.id)
    reasoner.prepare_for(dataset.id)

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(results_path, 'not found; starting with empty results.')
        results = {}

    for i, p in enumerate(dataset.problems):
        if max_problems is not None and i >= max_problems:
            break

        key = f'({dataset.id}, {p.id}, {reasoner.name()})'

        if key in results and not results[key]['error']:
            # print('Skipping', key)
            success.append(results[key]['correct'])
            continue

        try:
            prediction, reasoning = reasoner.predict_answer(p)
            error = None
            correct = (prediction == p.test_example.answer)
            print(key, 'success?', correct)
        except (Exception, RuntimeError) as e:
            print('Error:', e)
            correct = False
            raise
            error = str(e)
            prediction, reasoning = None, None

        success.append(correct)

        results_obj = {
            'dataset': dataset.id,
            'problem': p.id,
            'reasoner': reasoner.name(),
            'prediction': prediction,
            'answer': p.test_example.answer,
            'error': error,
            'reasoning': reasoning,
            'correct': correct
        }

        results[key] = results_obj
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    print(f'Accuracy: {100 * sum(success) / len(success):.2f}%')


# TODO: refactor 
def run_beerqa_experiments(max_problems=40):
    datasets = [
        # PrOntoQADataset.load('./prontoqa/1hop_random_seed19.json'),
    ]

    # fol_domain = domain.FirstOrderLogicDomain()
    # fol_completion_engine = PeanoCompletionEngine(
    #     fol_domain,
    #     fol_domain.start_derivation())

    reasoners = [
            #OpenAILanguageModelReasoner('text-davinci-003'),
    ]

    for r in reasoners:
        for ds in datasets:
            evaluate_reasoner('results.json', ds, r, max_problems)


# TODO: implement 
def run_qasperqa_experiments(max_problems=40):
    pass


if __name__ == '__main__':
    run_beerqa_experiments()
    # run_qasperqa_experiments()
