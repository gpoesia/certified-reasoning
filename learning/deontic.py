import os
import regex
import random
import argparse
import copy
import sys

sys.setrecursionlimit(10000)

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

    lines = text.split("\n")
    deontic_axioms = []
    theory_axioms = []
    for line in lines:
        if 'daxiom' in line:
            deontic_axioms.append(line)
        elif 'taxiom' in line:
            deontic_axioms.append(line)
            
    deontic_axioms = "\n".join(deontic_axioms)
    theory_axioms = "\n".join(theory_axioms)
    return deontic_axioms, theory_axioms

def copy_problem(problem, calendar):
    copied_problem = calendar.start_derivation(problem=problem.description, goal=None)
    copied_problem.universe = problem.universe.clone()
    return copied_problem


def dfs_search(cal_problem, calendar, max_depth):
    stack = [(cal_problem, 0, [])]
    visited = set()


    while stack:
        cur_problem, depth, path = stack.pop()
        if args.verbose:
            print(f' Depth: {depth} len stack: {len(stack)}')
        # if (depth, cur_problem.universe) in visited:
        #     continue
        # visited.add((depth, cur_problem.universe))

        # Check if the last action is an axiom related action
        if depth == max_depth:
            if args.verbose:
                print('Found axiom')
            return path


        actions = calendar.derivation_actions(cur_problem.universe)
        if depth < max_depth - 1:
            actions = [action for action in actions if 'taxiom' in action]
        elif depth == max_depth - 1:
            actions = [action for action in actions if 'daxiom' in action]

        for action in actions:
            temp_problem = copy_problem(cur_problem, calendar)

            if depth == 0:
                outcomes = temp_problem.universe.apply(action)
            else:
                outcomes = temp_problem.universe.apply_with(action, f'r{depth}')

            if len(outcomes) == 0:
                continue

            for outcome in outcomes:
                temp_problem_o = copy_problem(temp_problem, calendar)
                if args.verbose:
                    print(f'outcome: {outcome}')
                try: 
                    calendar.define(temp_problem_o.universe, f'r{depth + 1}', outcome)
                except:
                    continue
                stack.append((temp_problem_o, depth + 1, path + [outcome, action]))
    return None

def exhaustive_search(problem, n_hops, calendar):
    cal_problem = calendar.start_derivation(problem=problem, goal=None)
    result = dfs_search(cal_problem, calendar, n_hops)
    # result = dfs_search(cal_problem, calendar, 0, n_hops, [])
    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4",
                        help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--max_tokens', type=int, default=400)
    parser.add_argument('--n_hops', type=int, default=2)
    parser.add_argument('--n_problems', type=int, default=1)
    parser.add_argument('--domain', type=str, default="calendar")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use_context', action='store_true')
    parser.add_argument('--use_axioms', action='store_true')
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



    if args.use_context:
        context = contexts
    else:
        context_prompt = get_context_prompt(system_context_dom, example_context_dom)
        context = get_chat_response(context_prompt, args)
    if args.verbose:
        print(f"Context: {context}")

    axiom_prompt = get_axiom_prompt(system_axioms_dom, axiom_templates_dom, example_axioms, example_context_dom, context)

    if args.use_axioms:
        gen_deontic_axioms = deontic_axioms 
        gen_theory_axioms = theory_axioms
    else:
        axiom_response = get_chat_response(axiom_prompt, args)
        gen_deontic_axioms, gen_theory_axioms = get_axioms(axiom_response) 

#     gen_deontic_axioms = """let axiom1 : [('e : event) -> ('a : person) -> (free 'e 'a) -> (permissible (send_notification 'b 'f))].
# let axiom2 : [('e : event) -> ('g : group) -> (group_participant 'g 'e) -> (permissible (accept (group_invite 'g 'e)))].
# let axiom3 : [('e : event) -> ('a : person) -> (busy 'a 'e) -> (impermissible (reschedule_event 'e daily))].
# let axiom4 : [('e : event) -> ('a : person) -> (high 'a 'e) -> (obligatory (set_reminder (hours_before 'a 'e)))].
# let axiom5 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (delegate_event 'e 'a))].
# let axiom6 : [('e : event) -> ('a : person) -> (long 'e) -> (permissible (update_event 'e conference))].
# let axiom7 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (impermissible (remove_participant 'e 'g))].
# let axiom8 : [('e : event) -> ('a : person) -> (free 'a 'e) -> (obligatory (accept (individual_invite 'a 'e)))].
# let axiom9 : [('e : event) -> ('a : person) -> (tentative 'a 'e) -> (permissible (suggest_alternative_time 'a 'e))]."""

#     gen_theory_axioms = """
# let taxiom0 : [('e : event) -> ((individual_invite a1 'e): invite) -> (short 'e)].
# let taxiom1 : [('e : event) -> (daily 'e) -> (long 'e)].
# let taxiom2 : [('p : person) -> ('e : event) -> (low 'e 'p) -> (busy'e 'p)].
# let taxiom3 : [('p : person) -> ('e : event) -> (high 'e 'p) -> (free'e 'p)].
# let taxiom4: [('e : event) -> (weekly 'e) -> (high a2 'e)].
# let taxiom5: [('e : event) -> ('p : person) -> (high 'p 'e) -> (participant 'e 'p)]."""

    if args.verbose:
        print(f"Deontic Axioms: {gen_deontic_axioms}")
        print(f"Theory Axioms: {gen_theory_axioms}")

    # define the context
    problem = f"{context}\n{gen_deontic_axioms}{gen_theory_axioms}"

    # define the theory
    calendar = CalendarDomain('calendar.p')

    # sample an outcome
    sampled_outcomes = []

    
    # Execute the exhaustive search
    result = exhaustive_search(problem, n_hops, calendar)
    print("Exhaustive search result:", result)


    ### debug

    # convert context to a scenario - gpt-4

    # convert deontic_axioms to constraint - gpt-4

    # convert context + theory to a situation - gpt-4

    # convert outcome to a question - gpt-4
