import os
import regex
import random
import argparse

import openai

import peano
from deontic_domains.axiom_templates import *
from deontic_domains.prompts import *
from deontic_domains.calendar import CalendarDomain, contexts, deontic_axioms, theory_axioms

def get_chat_response(prompt_messages, args):
    response = openai.ChatCompletion.create(model=args.model, messages=prompt_messages,
                                                temperature=args.temperature,
                                                max_tokens=args.max_tokens)
    prediction = response['choices'][0]['message']['content']
    return prediction

def get_axioms(text):
    deontic_start = text.find("Deoontic Axioms:") + len("Deoontic Axioms:")
    deontic_end = text.find("Theory Axioms:")

    theory_start = text.find("Theory Axioms:") + len("Theory Axioms:")
    theory_end = len(text)

    deontic_axioms = text[deontic_start:deontic_end]
    theory_axioms = text[theory_start:theory_end]
    return deontic_axioms, theory_axioms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4",
                        help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=400)
    parser.add_argument('--n_hops', type=int, default=1)
    parser.add_argument('--n_problems', type=int, default=1)
    parser.add_argument('--domain', type=str, default="calendar")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    theory_file = f"../environment/theories/{args.domain}.p" 
    with open(theory_file, 'r') as f:
        theory = f.read()
    n_hops = args.n_hops
    n_problems = args.n_problems

    # sample context using gpt-4
    system_context_dom = system_context.format(theory=theory)
    example_context_dom = example_context.format(context=contexts)
    system_axioms_dom = system_axioms.format(theory=theory)
    axiom_templates_dom = axiom_templates.format(deontic_axioms=deontic_templates, theory_axioms=theory_templates)
    example_axioms = deontic_axioms.format(deontic_axiom=deontic_axioms, theory_axioms=theory_axioms)

    context_prompt = get_context_prompt(system_context_dom, example_context_dom)
    context = get_chat_response(context_prompt, args)
    if args.verbose:
        print(f"Context: {context}")

    axiom_prompt = get_axiom_prompt(system_axioms_dom, axiom_templates_dom, example_axioms, example_context_dom, context)

    axiom_response = get_chat_response(axiom_prompt, args)
    gen_deontic_axioms, gen_theory_axioms = get_axioms(axiom_response) 
    if args.verbose:
        print(f"Deontic Axioms: {gen_deontic_axioms}")
        print(f"Theory Axioms: {gen_theory_axioms}")

    # define the context
    problem = f"{context}\n{gen_deontic_axioms}{gen_theory_axioms}"

    # define the theory
    calendar = CalendarDomain('calendar.p')

    # sample an outcome
    sampled_outcomes = []

    cal_problem = calendar.start_derivation(problem=problem, goal=None)
    failures = 0
    while len(sampled_outcomes) < n_hops and failures < 3:
        actions = calendar.derivation_actions(cal_problem.universe)
        if len(sampled_outcomes) == n_hops:
            # final outcome has to be an axiom related outcome
            action = random.choice([a for a in actions if 'axiom' in a])
        else:
            action = random.choice(actions)
        if args.verbose:
            print("action", action)
        if len(sampled_outcomes) == 0:
            outcomes = cal_problem.universe.apply(action)
        else:
            # action, definition
            outcomes =  cal_problem.universe.apply_with(action, f'r{len(sampled_outcomes)}')
        print("outcomes", outcomes)
        if len(outcomes) == 0:
            sampled_outcomes = []
            cal_problem = calendar.start_derivation(problem=problem, goal=None)
            failures += 1
            if failures == 10:
                break
            if args.verbose:
                print("Restarting derivation")
            continue 
        outcome = random.choice(outcomes)
        sampled_outcomes.append(outcome)
        parent_args = outcome.generating_arguments()
        calendar.define(cal_problem.universe, f'r{len(sampled_outcomes)}', outcome)
    if args.verbose:
        print(sampled_outcomes)
    
    # actions = calendar.derivation_actions(cal_problem.universe)
    # action = random.choice([a for a in actions if regex.fullmatch('axiom\\d+', a)])
    # outcomes = cal_problem.universe.apply(action)
    # outcome = random.choice(outcomes)
    # sampled_outcomes.append(outcome)
    # calendar.define(cal_problem.universe, f'r{len(sampled_outcomes)}', outcome)
    # print(sampled_outcomes)


    # convert context to a scenario - gpt-4

    # convert deontic_axioms to constraint - gpt-4

    # convert context + theory to a situation - gpt-4

    # convert outcome to a question - gpt-4
