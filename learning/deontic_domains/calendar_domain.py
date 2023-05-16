from domain import DomainFromTheory, Problem
from typing import Optional

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

let org1 : (organizer e1 a1). 
let part1 : (participant e1 a2).
let rec1 : (daily e3).
let rec2 : (weekly e2).
let prio1 : (low a1 e2).
let cat1 : (meeting e3).
let inv1 : invite = (individual_invite a1 e1).
let inv2 : invite = (group_invite g1 e2)."""

deontic_axioms = """let daxiom1 : [('e : event) -> ('a : person) -> (free 'e 'a) -> (permissible (send_notification 'b 'f))].
let daxiom2 : [('e : event) -> ('g : group) -> (group_participant 'g 'e) -> (permissible (accept (group_invite 'g 'e)))].
let daxiom3 : [('e : event) -> ('a : person) -> (busy 'a 'e) -> (impermissible (reschedule_event 'e daily))].
let daxiom4 : [('e : event) -> ('a : person) -> (high 'a 'e) -> (obligatory (set_reminder (hours_before 'a 'e)))].
let daxiom5 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (delegate_event 'e 'a))].
let daxiom6 : [('e : event) -> ('a : person) -> (long 'e) -> (permissible (update_event 'e conference))].
let daxiom7 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (impermissible (remove_participant 'e 'g))].
let daxiom8 : [('e : event) -> ('a : person) -> (free 'a 'e) -> (obligatory (accept (individual_invite 'a 'e)))].
let daxiom9 : [('e : event) -> ('a : person) -> (tentative 'a 'e) -> (permissible (suggest_alternative_time 'a 'e))].
let daxiom10 : [('e : event) -> (private 'e) -> (obligatory (check_availability a1 'e))].
"""

theory_axioms = """let taxiom1 : [('e : event) -> ((individual_invite a1 'e): invite) -> (short 'e)].
let taxiom2 : [('e : event) -> (daily 'e) -> (long 'e)].
let taxiom3 : [('e : event) -> (long 'e) -> (private 'e)].
let taxiom4 : [('e : event) -> (private 'e) -> (group_participant 'e g1)].
let taxiom5 : [('g : group) -> ('e : event) -> (group_participant 'e 'g) -> (meeting 'e)].
let taxiom6: [('e : event) -> (meeting 'e) -> (high a2 'e)].
let taxiom7 : [('e : event) -> (meeting 'e) -> (organizer 'e a3)].
let taxiom8 : [('p : person) -> (organizer e1 'p) -> (high 'p e1)].
let taxiom9 : [('e : event) -> (high 'e a1) -> (daily 'e)].
let taxiom10 : [('e : event) -> (daily 'e) -> (days_before a2 'e)].
"""

domain_text = "Create a task for an AI assistant managing the calendar."