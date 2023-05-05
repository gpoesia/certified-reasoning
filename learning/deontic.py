import os
import peano
import regex
import random
from domain import DomainFromTheory, Problem
from typing import Optional
from completion import PeanoCompletionEngine

def get_domain(theory_file):
    with open(theory_file) as f:
        theory = f.read()
    derivation = peano.PyDerivation()
    derivation.incorporate(theory)
    return derivation

class CalendarDomain(DomainFromTheory):
    def __init__(self, theory, with_equality=False):
        super().__init__(theory, [])
        self._ignored_actions = {'eval', '='}.union(
            {'eq_refl', 'eq_symm', 'rewrite'}
            if not with_equality
            else set()
        )

    def generate_derivation(self, _seed: int):
        return self.start_derivation(None, None)

    def start_derivation(self, problem=None, goal=None):
        u = self.base_derivation.clone()
        if problem:
            u.incorporate(problem)
        domain_problem = Problem(u, problem, goal, self)
        return domain_problem

    def derivation_actions(self, d) -> list[str]:
        return list(set(d.actions()) - self._ignored_actions)

    @staticmethod
    def _negate(goal: str) -> str:
        if goal.startswith('(not '):
            return goal[len('(not '):-1]
        else:
            return f'(not {goal})'

    def derivation_done(self, problem: Problem) -> Optional[str]:
        'Find a proof of either the goal type or its negation.'

        # If no goal was set yet, not done.
        if problem.goal is None:
            return None
        for name, dtype, _, is_prop, _deps in problem.universe.state():
            if not is_prop:
                continue

            if dtype in (problem.goal):
                return name, dtype == problem.goal
        return None


if __name__ == "__main__":
    theory_file = "../environment/theories/deontic.p" 
    n_hops = 2
    n_problems = 1

    derivation = get_domain(theory_file)

    # sample using gpt-4
    context = """let a1 : person.
let a2 : person.
let a3 : person.

let g1 : group.
let g2 : group.

let e1 : event.
let e2 : event.
let e3 : event.
let e4 : event.
let e5 : event.

let inv1 : invite = (individual_invite a1 e1).

"""
    # fill templates with values
    deontic_axiom = """let axiom1 : [('e : event) -> (permissible (accept (individual_invite a1 'e)))].  
let axiom2 : [('e : event) -> ('g : group) -> (permissible (decline (group_invite 'g 'e)))].  
let axiom3 : [('e : event) -> (permissible (add_participant 'e a2 'e))].  
let axiom4 : [('e : event) -> ('a : person) -> (impermissible (remove_participant 'e 'a 'e))].  
let axiom5 : [('ea : event) -> ('eb : event) -> (obligatory (update_event 'ea 'eb 'ea))].  
let axiom6 : [('e : event) -> (optional (change_visibility 'e private 'e))].  
let axiom7 : [('e : event) -> ('r : reminder) -> (notoptional (set_reminder 'e 'r 'e))].  
let axiom8 : [('e : event) -> (obligatory (cancel_event 'e)) -> (impermissible (reschedule_event 'e short 'e))].  
let axiom9 : [('e : event) -> (permissible (request_event_update a3 e4))].  
let axiom10 : [('e : event) -> (tentative a1 'e) -> (optional (delegate_event 'e a2))].  
let axiom11 : [('a: person) -> ('e : event) -> (short 'e) -> (permissible (accept (individual_invite 'a 'e)))].
"""
    theory_axiom = "let axiom0 : [('e : event) -> ((individual_invite a1 'e): invite) -> (short 'e)]."
    # define the context
    problem = f"{context}\n{deontic_axiom}\n{theory_axiom}"

    # define the theory
    calendar = CalendarDomain('deontic.p')

    # sample an outcome
    sampled_outcomes = []

    cal_problem = calendar.start_derivation(problem=problem, goal=None)
    while len(sampled_outcomes) < n_hops:
        actions = calendar.derivation_actions(cal_problem.universe)
        action = random.choice(actions)
        if len(sampled_outcomes) == 0:
            outcomes = cal_problem.universe.apply_with('axiom0', 'inv1')
        else:
            # action, definition
            outcomes =  cal_problem.universe.apply_with('axiom11', f'r{len(sampled_outcomes)}')
            print(outcomes)

        outcome = random.choice(outcomes)
        sampled_outcomes.append(outcome)
        parent_args = outcome.generating_arguments()
        calendar.define(cal_problem.universe, f'r{len(sampled_outcomes)}', outcome)
    print(sampled_outcomes)
    
    # final outcome has to be an axiom related outcome
    actions = calendar.derivation_actions(cal_problem.universe)
    action = random.choice([a for a in actions if regex.fullmatch('axiom\\d+', a)])
    outcomes = cal_problem.universe.apply(action)
    outcome = random.choice(outcomes)
    sampled_outcomes.append(outcome)
    calendar.define(cal_problem.universe, f'r{len(sampled_outcomes)}', outcome)
    print(sampled_outcomes)


    # convert context to a scenario - gpt-4

    # convert deontic_axioms to constraint - gpt-4

    # convert context + theory to a situation - gpt-4

    # convert outcome to a question - gpt-4

    
    
    
    
    
    
    
    
    

    # universe = calendar.start_derivation().universe.clone()
    # universe.incorporate(problem)
    # arrows = set(calendar.derivation_actions(universe)).union(initial_actions)
    # choices = []

    # for a in arrows:
    #     if a in initial_actions or regex.fullmatch('axiom\\d+', a):
    #         choices.extend(calendar.apply(a, universe))

    
    # print(choices)
    # choice = random.choice(choices)
    # universe.apply(choice)

    
    

    # derivation.incorporate(problem)
    # we want to generate a random problem by sampling a radom
    # path of actions/ steps from the theory
    # for i in range(1):
    #     base = random.choice(derivation.actions())
    #     print(base)
    #     actions = derivation.apply(base)
    #     # update the theory with the new action
    #     if len(actions) > 0:
    #         action = random.choice(actions)
    #         action_str = str(action)
    #         action_str = action_str.split(':')[0][1:-1]
    #         print(action)
    #         print(action_str)
    
    #         derivation.incorporate(action_str)
