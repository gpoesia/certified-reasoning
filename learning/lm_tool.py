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
import copy
from typing import List, Dict

import openai
import transformers

import domain
import tactics
import util
from completion_engine import CompletionEngine
from language_model import OpenAIModel, LanguageModel, download_or_use_cached, filter_maximal_tokens
from synchromesh import predict_constrained


from completion import PeanoCompletionEngine


class OpenAIChatModel(LanguageModel):
    # add this class to the Synchromesh lm module
    def __init__(self, model: str, prompt_template: List[Dict[str,str]], api_key: str = None,
                 temperature: float = 0.0, top_p: float = 1.0, best_of: int = 1,
                 before_prediction_hook=lambda: None) -> None:
        super().__init__()

        if api_key:
            openai.api_key = api_key

        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.best_of = best_of
        self._before_prediction_hook = before_prediction_hook

        # for gpt series of models
        if model.startswith("text") or model.startswith("gpt"):
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        elif model.startswith("code"):
            self.tokenizer = transformers.GPT2Tokenizer(
                vocab_file=download_or_use_cached(
                    "https://huggingface.co/SaulLu/codex-like-tokenizer/raw/main/vocab.json",
                    '.codex-vocab.json'),
                merges_file=download_or_use_cached(
                    "https://huggingface.co/SaulLu/codex-like-tokenizer/raw/main/merges.txt",
                    '.codex-merges.txt')
                )
            

        # self.vocab is a list of readable token strings (e.g., ' hello' and '\n')
        # sorted by their token IDs (so self.vocab[0] is the first token, etc).
        self.vocab = [v for k, v in
                      sorted([(t_id, self.tokenizer.decode([t_id]))
                              for _, t_id in self.tokenizer.get_vocab().items()])]

    def tokenize(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    def vocabulary(self) -> list[str]:
        # sort keys by value, then return the keys
        return self.vocab
    
    def predict_token(self, prefix: str, valid_tokens: list[int], top_k: int = 1) -> tuple[list[int], list[float]]:
        # change bias of valid tokens to make them more likely
        # bias can only be set for 300 tokens at a time
        assert top_k == 1, "Chat models do not support logprobs"
        prompt = copy.deepcopy(self.prompt_template)
        assert prompt[-1]['role'] == 'assistant', "last message must be from the assistant"
        prompt[-1]['content'] += prefix

        # Only keep tokens that cannot be extended. This is crucial, because
        # the model has *never* seen a sequence of non-maximal tokens in its
        # input, and if we force it to output a sequence of maximal tokens,
        # a logit bias is often not enough to constrain it (it outputs near
        # zero probability for the valid tokens even after adding 100 to
        # the logits).
        #
        # Longer explanation:
        # Suppose we want to force the model to output
        # the number 20302. This would be BPE-tokenized as 20 | 302.
        # Suppose we let the model output '2' alone. We succeed at that,
        # but now we need the model to output the token '0' alone. Here
        # we hit the problem: the sequence of tokens '2' followed by '0'
        # was seen exactly 0 times during training, since the tokenizer
        # will never emit this sequence (it will instead use the '20' token).
        # Hence, the model puts near 0 probability to predicting '0'
        # after predicting '2'. The solution is to only let the model
        # output maximal valid tokens, avoiding this issue.
        valid_tokens = filter_maximal_tokens(valid_tokens, self.tokenizer)

        if len(valid_tokens) == 1:
            return valid_tokens, [0.0]

        # select shortest valid tokens if valid tokens are less than 300
        if len(valid_tokens) >= 299:
            token_lens = [len(self.get_token(i)) for i in valid_tokens]
            # sort valid tokens by length
            valid_tokens = [x for _, x in sorted(zip(token_lens, valid_tokens))]
            valid_tokens = valid_tokens[:299*4-1]

        valid_bias = {k: 100 for k in valid_tokens[:299]}
        # add a negative bias for the stop token
        valid_bias[50256] = -100
        self._before_prediction_hook()
        response = openai.ChatCompletion.create(model=self.model, messages=prompt,
                                            temperature=self.temperature, top_p=self.top_p,
                                            max_tokens=1, logit_bias=valid_bias)
        # chat models do not return logprobs
        predictions, probabilities = [response['choices'][0]['message']['content']], [0.0]
        return predictions, probabilities

    def predict_unconstrained(self, prefix, max_tokens, stop=None):
        # here the last message must be from the assistant
        prompt = copy.deepcopy(self.prompt_template)
        assert prompt[-1]['role'] == 'assistant', "last message must be from the assistant"
        prompt[-1]['content'] += prefix
        self._before_prediction_hook()
        response = openai.ChatCompletion.create(model=self.model, messages=prompt,
                                            temperature=self.temperature, top_p=self.top_p,
                                            logit_bias={50256: -100},
                                            max_tokens=max_tokens, stop=stop)
        return response['choices'][0]['message']['content']


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


class PeanoChatLMReasoner:
    def __init__(self, completion_engine: CompletionEngine,
                 prompt_file: str,
                 model: str,
                 temperature: float = 0.0):
        self._completion_engine = completion_engine
        self._model = model
        self._temperature = temperature
        self._separator = '###'

        with open(prompt_file, 'r') as f:
            self._prompt = json.load(f)

    def name(self) -> str:
        return f'peano-chat-{self._model}'

    def _format_problem(self, problem) -> List[Dict[str, str]]:
        context = f' '.join(f'{i+1}- {sentence}'
                            for i, sentence in enumerate(problem.test_example.theory))
        query = problem.test_example.query
        chat_problem = [{"role": "user", "content": f"Context: {context}\nQuery: {query.strip()}"},
                        {"role": "assistant", "content": ""}]
        return chat_problem
    
    def predict_answer(self, problem: PrOntoQAProblem) -> bool:

        test = self._format_problem(problem)
        prompt_messages = self._prompt + test
        lm = OpenAIChatModel(self._model,
                         prompt_messages,
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
        # PeanoLMReasoner(fol_completion_engine,
        #                'prompts/peano_prontoqa_long_prompt',
        #                'text-davinci-003'),
        PeanoChatLMReasoner(fol_completion_engine,
                       'prompts/peano_chat_prontoqa_long_prompt.json',
                       'gpt-3.5-turbo')
        # OpenAIChatModelReasoner('gpt-3.5-turbo'),
        # OpenAIChatModelReasoner('gpt-4'),
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
    run_prontoqa_experiments()
    # run_math_experiments()
