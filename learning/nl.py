#!/usr/bin/env python3

'''
Reasoning in natural language (PrOntoQA dataset).
'''

import datetime
import json
import time
from dataclasses import dataclass
import random
import re

import openai

import domain
import tactics
import util
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
            wait = (self._min_interval - (now - self._last_request)).total_seconds()
            if wait > 0:
                print('wait', wait)
            time.sleep(max(0, wait))

        self._last_request = datetime.datetime.now()


rate_limiter = RateLimiter(20)


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
                    answer=example_obj['answer'])

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


@dataclass
class MathDataset:
    id: str
    problems: list[PrOntoQAProblem]

    @staticmethod
    def generate(n_problems: int, seed: int):
        random.seed(seed)

        def generate_subst_eval_problem_1():
            f = random.randint(-3, 3)
            g = random.randint(-10, 10)
            h = random.randint(-10, 10)

            c1 = random.randint(0, 15)
            c2 = random.randint(0, 9)

            theory = [f'Let x = f^3 + {c1}g - {c2}h.',
                      f'Suppose f = {f}, g = {g} and h = {h}.']

            query = 'Find the value of x.'
            answer = str(f**3 + c1*g - c2*h)
            chain_of_thought = [
                f'Substituting f, we get x = {f}*{f}*{f} + {c1}g - {c2}h.',
                f'Substituting g, we get x = {f}*{f}*{f} + {c1}{g} - {c2}h.',
                f'Substituting h, we get x = {f}*{f}*{f} + {c1}{g} - {c2}{h}.',
                'We can now evaluate the expression step-by-step to get the value of x.',
                f'x = {f*f}*{f} + {c1}{g} - {c2}{h}.',
                f'x = {f*f*f} + {c1}{g} - {c2}{h}.',
                f'x = {f*f*f} + {c1*g} - {c2}{h}.',
                f'x = {f*f*f + c1*g} - {c2}{h}.',
                f'x = {f*f*f + c1*g} - {c2*h}.',
                f'x = {f*f*f + c1*g - c2*h}.',
            ]

            return PrOntoQAExample(theory, query, chain_of_thought, answer)

        def generate_subst_eval_problem_2():
            a = random.randint(1, 5) * random.choice([-1, 1])
            b = random.randint(0, 20)
            c = a * random.randint(-8, 8)

            d = random.randint(1, 10)
            k = d * random.randint(1, 10)

            theory = [f'Let x = c/{a} - {b} + {k}/d',
                      f'Suppose c = {c} and d = {d}.']

            query = 'Find the value of x.'
            answer = str(c//a - b + k//d)

            chain_of_thought = [
                f'Substituting c, we get x = {c}/{a} - {b} + {k}/d.',
                f'Substituting d, we get x = {c}/{a} - {b} + {k}/{d}.',
                'We can now evaluate the expression step-by-step to get the value of x.',
                f'x = {c//a} - {b} + {k}/{d}.'
                f'x = {c//a - b} + {k}/{d}.'
                f'x = {c//a - b} + {k//d}.'
                f'x = {c//a - b + k//d}.'
            ]

            return PrOntoQAExample(theory, query, chain_of_thought, answer)

        def generate_subst_eval_problem_3():
            a = random.randint(-5, 5)
            j = random.randint(0, 10)
            k = random.randint(0, 4)

            theory = [f'Let x = {a} + jk + k^3.'
                      f'Suppose j = {j} and k = {k}.']

            query = 'Find the value of x.'
            answer = str(a + j*k + k**3)

            chain_of_thought = [
                f'Substituting j, we get x = {a} + {j}k + k^3.',
                f'Substituting k, we get x = {a} + {j}*{k} + {k}*{k}*{k}.',
                'We can now evaluate the expression step-by-step to get the value of x.',
                f'x = {a} + {j*k} + {k}*{k}*{k}',
                f'x = {a + j*k} + {k}*{k}*{k}',
                f'x = {a + j*k} + {k*k}*{k}',
                f'x = {a + j*k} + {k*k*k}',
                f'x = {a + j*k + k*k*k}'
            ]

            return PrOntoQAExample(theory, query, chain_of_thought, answer)

        def generate_subst_eval_problem_4():
            g = random.randint(1, 8)
            c1 = g * random.randint(1, 15)
            c2 = random.randint(1, 15)
            h = random.randint(-10, 10)
            c3 = random.randint(0, 20)

            theory = [f'Let x = {c1}/g + {c2}h + {c3}.'
                      f'Suppose g = {g} and h = {h}.']

            query = 'Find the value of x.'
            answer = str(c1//g + c2*h + c3)

            chain_of_thought = [
                f'Substituting g, we get x = {c1}/{g} + {c2}h + {c3}.',
                f'Substituting h, we get x = {c1}/{g} + {c2}{h} + {c3}.',
                'We can now evaluate the expression step-by-step to get the value of x.',
                f'x = {c1//g} + {c2}{h} + {c3}.',
                f'x = {c1//g} + {c2*h} + {c3}.',
                f'x = {c1//g + c2*h} + {c3}.',
                f'x = {c1//g + c2*h + c3}.',
            ]

            return PrOntoQAExample(theory, query, chain_of_thought, answer)

        problems = []

        for i in range(n_problems):
            train_examples = [generate_subst_eval_problem_1(),
                              generate_subst_eval_problem_2()]

            test_example = random.choice(
                [generate_subst_eval_problem_1, generate_subst_eval_problem_2,
                 generate_subst_eval_problem_3, generate_subst_eval_problem_4])()

            problems.append(
                PrOntoQAProblem(f'problem{i}', train_examples, test_example)
                )

        return PrOntoQADataset(f'subst-eval-{n_problems}S{seed}', problems)


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
        answer = re.search('Answer: (.+)$', response_str)
        return answer and answer.groups()[-1].strip(), response_str


class OpenAIChatModelReasoner:
    def __init__(self, model: str, temperature: float = 0.0):
        self._model = model
        self._temperature = temperature
        self._separator = '###'

    def name(self) -> str:
        return self._model

    def _format_example(self, example: PrOntoQAExample,
                        index: int, is_test: bool):
        lines = []

        messages = []

        lines.append(f'Problem #{index + 1}')
        lines.append(f'Context: {" ".join(example.theory)}')
        lines.append(f'Query: {example.query}')

        messages.append({'role': 'user', 'content': '\n'.join(lines)})

        if not is_test:
            lines = []
            lines.append(f'Reasoning: {" ".join(example.chain_of_thought)}')
            lines.append(f'Answer: {str(example.answer)}')
            messages.append({'role': 'assistant', 'content': '\n'.join(lines)})

        return messages

    def predict_answer(self, problem: PrOntoQAProblem) -> bool:
        in_context = [m
                      for i, e in enumerate(problem.train_examples[:3])
                      for m in self._format_example(e, i, False)]

        test = self._format_example(problem.test_example, len(in_context), True)

        prompt = ([{"role": "system",
                    "content": "You are an AI reasoner that always follows the specified format."}]
                  + in_context + test)

        rate_limiter.wait()
        response = openai.ChatCompletion.create(model=self._model,
                                                messages=prompt,
                                                max_tokens=500,
                                                stop=self._separator)

        response_str = response.choices[0]['message']['content']
        answer = re.search('Answer: (.+)$', response_str)
        return answer and answer.groups()[-1].strip(), response_str


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
                         before_prediction_hook=rate_limiter.wait,
                         )

        response = predict_constrained(self._completion_engine, lm, batch_size=500,
                                       stop_tokens=[self._separator])
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

        if key in {
                '(subst-eval-30St, problem2, peano-text-davinci-003)',
                '(subst-eval-30St, problem3, peano-text-davinci-003)',
                '(subst-eval-30St, problem7, peano-text-davinci-003)',
                '(subst-eval-30St, problem9, peano-text-davinci-003)',
                '(subst-eval-30St, problem10, peano-text-davinci-003)',
                '(subst-eval-30St, problem11, peano-text-davinci-003)',
        }:
            continue

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


def run_prontoqa_experiments():
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
        # OpenAILanguageModelReasoner('text-davinci-003'),
        #PeanoLMReasoner(fol_completion_engine,
        #                'prompts/peano_prontoqa_long_prompt',
        #                'text-davinci-003'),
        OpenAIChatModelReasoner('gpt-3.5-turbo'),
        OpenAIChatModelReasoner('gpt-4'),
        # PeanoLMReasoner(fol_completion_engine,
        #                'prompts/peano_prontoqa_long_prompt',
        #                'code-davinci-002'),
    ]

    for ds in datasets:
        for r in reasoners:
            evaluate_reasoner('results.json', ds, r)


def run_math_experiments():
    datasets = [
        MathDataset.generate(30, 't')
    ]

    ka_domain = domain.AlgebraDomain()
    ka_domain.load_tactics([
        tactics.Tactic(
            'eval_rewrite',
            [
                tactics.Step(['eval'], ['?a'], '?0'),
                tactics.Step(['rewrite'], ['?0', '?b@*'], '?1'),
            ]
        )
    ])

    completion_engine = PeanoCompletionEngine(
        ka_domain,
        ka_domain.start_derivation(),
        util.format_infix
        )

    reasoners = [
        # OpenAIChatModelReasoner('gpt-4')
        OpenAIChatModelReasoner('gpt-3.5-turbo'),
        OpenAILanguageModelReasoner('text-davinci-003'),
        # PeanoLMReasoner(completion_engine,
        #                'prompts/peano_substeval_prompt',
        #                'text-davinci-003'),
    ]

    for ds in datasets:
        for r in reasoners:
            evaluate_reasoner('math_results.json', ds, r)


if __name__ == '__main__':
    # run_prontoqa_experiments()
    run_math_experiments()
