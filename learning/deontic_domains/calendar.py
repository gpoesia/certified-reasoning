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

let e1 : event.
let e2 : event.
let e3 : event.

let r1 : (hours_before a1 e1).
let r2 : (days_before a2 e2).

let dur1 : (short e1).
let dur2 : (long e2).

let p1 : (high a1 e1).
let p2 : (low a2 e2).

let rec1 : (daily e1).
let rec2 : (weekly e2).
let rec3 : (yearly e3).

let cat1 : (meeting e1).
let cat2 : (conference e2).
let cat3 : (social e3).

let vis1 : (public e1).
let vis2 : (private e2).
let vis3 : (confidential e3).

let inv1 : invite = (individual_invite a1 e1).
let inv2 : invite = (group_invite g1 e2)."""

deontic_axioms = """let axiom1 : [('e : event) -> ('a : person) -> (free 'e 'a) -> (permissible (send_notification 'b 'f))].
let axiom2 : [('e : event) -> ('g : group) -> (group_participant 'g 'e) -> (permissible (accept (group_invite 'g 'e)))].
let axiom3 : [('e : event) -> ('a : person) -> (busy 'a 'e) -> (impermissible (reschedule_event 'e daily))].
let axiom4 : [('e : event) -> ('a : person) -> (high 'a 'e) -> (obligatory (set_reminder (hours_before 'a 'e)))].
let axiom5 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (delegate_event 'e 'a))].
let axiom6 : [('e : event) -> ('a : person) -> (long 'e) -> (permissible (update_event 'e conference))].
let axiom7 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (impermissible (remove_participant 'e 'g))].
let axiom8 : [('e : event) -> ('a : person) -> (free 'a 'e) -> (obligatory (accept (individual_invite 'a 'e)))].
let axiom9 : [('e : event) -> ('a : person) -> (tentative 'a 'e) -> (permissible (suggest_alternative_time 'a 'e))]."""

theory_axioms = """let taxiom1 : [('e : event) -> ((individual_invite a1 'e): invite) -> (short 'e)].
let taxiom2 : [('e : event) -> (daily 'e) -> (long 'e)].
let taxiom3 : [('p : person) -> ('e : event) -> (low 'e 'p) -> (busy'e 'p)].
let taxiom4 : [('p : person) -> ('e : event) -> (high 'e 'p) -> (free'e 'p)].
let taxiom5: [('e : event) -> (weekly 'e) -> (high a2 'e)]."""