from domain import DomainFromTheory, Problem
from typing import Optional
from completion import PeanoCompletionEngine

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

contexts = """let a1 : person.
let a2 : person.
let a3 : person.

let g1 : group.
let g2 : group.

let e1 : event.
let e2 : event.
let e3 : event.
let e4 : event.
let e5 : event.
let inv1 : invite = (individual_invite a1 e1)."""

deontic_axioms = """let axiom1 : [('e : event) -> ('p : person) -> (participant e p) -> (permissible (accept (individual_invite p e)))].
let axiom2 : [('e : event) -> ('p : person) -> (organizer e p) -> (permissible (cancel_event e))].
let axiom3 : [('e : event) -> ('p : person) -> (low p e) -> (permissible (set_reminder (days_before p e)))].
let axiom4 : [('e : event) -> ('p : person) -> (organizer e p) -> (obligatory (update_event e public))].
let axiom5 : [('e : event) -> ('p : person) -> (participant e p) -> (impermissible (suggest_alternative_time p e))]."""

theory_axioms = """let taxiom1 : [('e : event) -> (participant e a1) -> ('p : person) -> (not (organizer e p))].
let taxiom2 : [('e : event) -> ('p : person) -> (high p e) -> (organizer e p)].
let taxiom3 : [('e : event) -> (public e) -> (not (confidential e))].
let taxiom4 : [('e : event) -> (private e) -> ('p : person) -> (not (high p e))].
let taxiom5 : [('e : event) -> (busy a1 e) -> ('p : person) -> (tentative p e)]."""