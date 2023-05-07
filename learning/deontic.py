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
        if 'exists' in line or 'forall' in line:
            continue
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
        random.shuffle(actions)
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

            random.shuffle(outcomes)
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
    parser.add_argument('--temperature', type=float, default=0.6)
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
        # manual intervention for getting hops
        context = """let b1 : person.
let b2 : person.
let b3 : person.
let g2 : group.
let f1 : event.
let f2 : event.
let f3 : event.

let org2 : (organizer f1 b1).
let part2 : (participant f1 b3).

let vis1 : (public f1).
let vis2 : (confidential f3).
let dur1 : (short f2).
let cat2 : (conference f1).
let cat3 : (social f3).
let rec3 : (monthly f2).

let reminder1 : reminder = (hours_before b1 f1).
let reminder2 : reminder = (days_before b2 f2).
let inv3 : invite = (individual_invite b1 f1).
"""
    else:
        context_prompt = get_context_prompt(system_context_dom, example_context_dom)
        context = get_chat_response(context_prompt, args)
    if args.verbose:
        print(f"Context: {context}")

    axiom_prompt = get_axiom_prompt(system_axioms_dom, axiom_templates_dom, example_axioms, example_context_dom, context)

    if args.use_axioms:
        # manual intervention for getting n hops
        gen_deontic_axioms = """let daxiom11 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (accept (individual_invite 'b 'f)))].
let daxiom12 : [('f : event) -> ('g : group) -> (group_participant 'g 'f) -> (permissible (send_notification (group_invite 'g 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (check_availability 'b 'f))].
let daxiom14 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let daxiom15 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (obligatory (update_event 'f social))].
let daxiom16 : [('f : event) -> ('b : person) -> (long 'f) -> (permissible (set_reminder (days_before 'b 'f)))].
let daxiom17 : [('f : event) -> ('g : group) -> (group_participant 'f 'g) -> (permissible (add_participant 'f 'b))].
let daxiom18 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (reschedule_event 'f daily))].
let daxiom19 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (suggest_alternative_time 'b 'f))].
let daxiom20 : [('f : event) -> (long 'f) -> (permissible (change_visibility 'f confidential))].
"""
        gen_theory_axioms = """
let taxiom1 : [('f : event) -> (social 'f) -> (public 'f)].
let taxiom2 : [('b : person) -> ('f : event) -> (busy 'b 'f) -> (low 'b 'f)].
let taxiom3 : [('f : event) -> (public 'f) -> (long 'f)].
let taxiom4 : [('f : event) -> (long 'f) -> (free b3 'f)].
let taxiom5 : [('b : person) -> ('f : event) -> (organizer 'b 'f) -> (confidential 'f)].
let taxiom6 : [('b : person) -> ('f : event) -> (free 'b 'f) -> (low 'b 'f)].
let taxiom7 : [('f : event) -> (short 'f) -> (conference 'f)].
let taxiom8 : [('b : person) -> ('f : event) -> (participant 'b 'f) -> (daily 'f)].
let taxiom10 : [('b : person) -> ('f : event) -> (busy 'b 'f) -> (yearly 'f)].
"""
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

    # store the problem, theory, context, outcome, and solution
    # check files to find the next problem id
    if result is not None:
        file_num = len([f for f in os.listdir("deontic_domains/") if f.endswith(".p") and f"calendar_problem_{args.n_hops}" in f])
        
        # save the problem as problem_00.p
        with open(f"deontic_domains/calendar_problem_{args.n_hops}_{file_num:02d}.p", 'w') as f:
            f.write(problem)
            f.write("\nResult:\n")
            result_str = '\n'.join([str(r) for r in result[::2]])
            f.write(result_str)


    # TODOS:
    # convert context to a scenario - gpt-4
    # convert deontic_axioms to constraint - gpt-4
    # convert context + theory to a situation - gpt-4
    # convert outcome to a question - gpt-4
