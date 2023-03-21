#!/usr/bin/env python3

'''
Reasoning in natural language (PrOntoQA dataset).
'''

import datetime
import json
import time
from dataclasses import dataclass

import openai

import domain
from completion_engine import CompletionEngine
from language_model import OpenAIModel
from synchromesh import predict_constrained

from completion import PeanoCompletionEngine



@dataclass
class PrOntoQAExample:
    theory: list[str]
    query: list[str]
    chain_of_thought: list[str]
    answer: bool


@dataclass
class PrOntoQAProblem:
    id: str
    train_examples: list[PrOntoQAExample]
    test_example: PrOntoQAExample


class RateLimiter:
    def __init__(self, rpm: int = 20):
        self._min_interval = datetime.timedelta(seconds = 60 / rpm)
        self._last_request = datetime.datetime.now()

    def wait(self):
        'Wait enough to not hit the rate limit.'
        now = datetime.datetime.now()

        if self._last_request is not None:
            wait = (self._min_interval - (self._last_request - now)).total_seconds()
            time.sleep(max(0, wait))

        self._last_request = datetime.datetime.now()


rate_limiter = RateLimiter()


def _split_question(question: str):
    return [s + '.' for s in question.rstrip('.').split('. ')]

@dataclass
class PrOntoQADataset:
    id: str
    problems: list[PrOntoQAProblem]

    @staticmethod
    def load(path: str) -> 'PrOntoQADataset':
        with open(path, 'rb') as f:
            data = json.load(f)

        problems = []

        for problem_id, problem_obj in data.items():
            train_examples = []
            test_example = None

            for example_id, example_obj in problem_obj.items():
                example = PrOntoQAExample(
                    theory=_split_question(example_obj['question']),
                    query=example_obj['query'],
                    chain_of_thought=example_obj['chain_of_thought'],
                    answer=example_obj['answer'] == 'True')

                if example_id == 'test_example':
                    test_example = example
                else:
                    train_examples.append(example)

            problems.append(PrOntoQAProblem(
                id=problem_id,
                train_examples=train_examples,
                test_example=test_example,
            ))

        return PrOntoQADataset(id=path, problems=problems)


class NaturalLanguageReasoner:
    def predict_answer(self, problem: PrOntoQAProblem) -> bool:
        raise NotImplementedError


class OpenAILanguageModelReasoner:
    def __init__(self, model: str, temperature: float = 0.0):
        self._model = model
        self._temperature = temperature
        self._separator = '###'

    def name(self) -> str:
        return self._model

    def _format_example(self, example: PrOntoQAExample,
                        index: int, is_test: bool):
        lines = []

        lines.append(f'Problem #{index + 1}')
        lines.append(f'Context: {" ".join(example.theory)}')
        lines.append(f'Query: {example.query}')

        if is_test:
            lines.append('Reasoning:')
        else:
            lines.append(f'Reasoning: {" ".join(example.chain_of_thought)}')
            lines.append(f'Answer: {str(example.answer)}')

        return '\n'.join(lines)

    def predict_answer(self, problem: PrOntoQAProblem) -> bool:
        in_context = [self._format_example(e, i, False)
                      for i, e in enumerate(problem.train_examples[:3])]
        test = self._format_example(problem.test_example, len(in_context), True)

        prompt = f'\n{self._separator}\n'.join(in_context + [test])

        rate_limiter.wait()
        response = openai.Completion.create(model=self._model,
                                            prompt=prompt,
                                            temperature=self._temperature,
                                            max_tokens=500,
                                            stop=self._separator)

        response_str = response.choices[0].text

        answer_true = 'Answer: True' in response_str
        answer_false = 'Answer: False' in response_str
        some_answer = answer_true or answer_false
        return answer_true if some_answer else None, response_str

class PeanoLMReasoner:
    def __init__(self, completion_engine: CompletionEngine,
                 prompt_file: str,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._separator = '###'


        with open(prompt_file, 'r') as f:
            self._prompt = f.read().strip()

    def name(self) -> str:
        return f'peano-{self._model}'

    def _format_problem(self, problem) -> str:
        context = f' '.join(f'{i+1}- {sentence}'
                            for i, sentence in enumerate(problem.test_example.theory))
        query = problem.test_example.query
        return f'Context: {context}\nQuery: {query.strip()}'

    def predict_answer(self, problem: PrOntoQAProblem) -> bool:
        prompt = f'{self._prompt}\n{self._format_problem(problem)}'

        lm = OpenAIModel(self._model,
                         prompt,
                         temperature=self._temperature,
                         before_prediction_hook=rate_limiter.wait)

        response = predict_constrained(self._completion_engine, lm, batch_size=500)
        done, answer = self._completion_engine.is_complete(response)
        assert done

        return answer, response

def evaluate_reasoner(results_path: str,
                      dataset: PrOntoQADataset,
                      reasoner: NaturalLanguageReasoner):
    success = []

    print('Evaluating', reasoner.name(), 'on', dataset.id)

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(results_path, 'not found; starting with empty results.')
        results = {}

    for p in dataset.problems:
        key = f'({dataset.id}, {p.id}, {reasoner.name()})'

        if key in results:
            print('Skipping', key)
            continue

        try:
            prediction, reasoning = reasoner.predict_answer(p)
            error = None
            correct = (prediction == p.test_example.answer)
            print(key, 'success?', correct)
        except Exception as e:
            print('Error:', e)
            correct = False
            error = str(e)
            prediction, reasoning = None, None

        success.append(correct)

        results_obj = {
            'dataset': dataset.id,
            'problem': p.id,
            'reasoner': reasoner.id,
            'prediction': prediction,
            'error': error,
            'reasoning': reasoning,
            'correct': correct
        }

        results[key] = results_obj
        with open(results_path, 'rb') as f:
            json.dump(results, f, indent=4)

    print(f'Accuracy: {100 * sum(success) / len(success):.2f}%')


if __name__ == '__main__':
    datasets = [
        PrOntoQADataset.load('./prontoqa/1hop_random.json'),
        PrOntoQADataset.load('./prontoqa/2hop_random.json'),
        PrOntoQADataset.load('./prontoqa/3hop_random.json'),
        PrOntoQADataset.load('./prontoqa/4hop_random.json'),
        PrOntoQADataset.load('./prontoqa/5hop_random.json')
    ]

    fol_domain = domain.FirstOrderLogicDomain()
    fol_completion_engine = PeanoCompletionEngine(
        fol_domain,
        fol_domain.start_derivation())

    reasoners = [
        #OpenAILanguageModelReasoner('text-davinci-003')
        #OpenAILanguageModelReasoner('gpt-4'),
        OpenAILanguageModelReasoner('text-davinci-003'),
        PeanoLMReasoner(fol_completion_engine,
                        'prompts/peano_prontoqa_long_prompt',
                        'text-davinci-003'),
        OpenAILanguageModelReasoner('code-davinci-002'),
        PeanoLMReasoner(fol_completion_engine,
                        'prompts/peano_prontoqa_long_prompt',
                        'code-davinci-002'),
    ]

    for ds in datasets:
        for r in reasoners:
            evaluate_reasoner('results.json', ds, r)
