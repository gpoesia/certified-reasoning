from domain import DomainFromTheory, Problem
from typing import Optional

class TriageDomain(DomainFromTheory):
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

contexts = """let h1 : healthcare_professional.
let h2 : healthcare_professional.
let h3 : healthcare_professional.
let p1 : patient.
let p2 : patient.

let t1 : treatment = medication p1.
let t2 : treatment = surgery p2.

let res1 : (high_resources h1).
let res2 : (low_resources h2).
let stab1 : (stable_vitals p1).
let crit1 : (critical_condition p2).

let cons1 : (conscious p1).
let cons2 : (unconscious p2).
let mob1 : (mobile p1).
let mob2 : (immobile p2).

let eq1 : equipment.
let eq2 : equipment.

let reh1 : (initiate_rehabilitation h1 p1).
let pall1 : (provide_palliative_care h2 p2).
let care1 : (assigned_care h1 p1).
let care2 : (not_assigned_care h2 p2).
"""

deontic_axioms = """let daxiom1 : [('h : healthcare_professional) -> ('p : patient) -> (critical_condition 'p) -> (obligatory (assign_red 'h 'p))].
let daxiom2 : [('h : healthcare_professional) -> ('p : patient) -> (stable_vitals 'p) -> (obligatory (assign_green 'h 'p))].
let daxiom3 : [('h : healthcare_professional) -> (unconscious p1) -> (obligatory (perform_examination 'h p1))].
let daxiom4 : [('h : healthcare_professional) -> ('p : patient) -> (immobile 'p) -> (not (obligatory (create_treatment_plan 'h 'p)))].
let daxiom5 : [('h : healthcare_professional) -> (high_resources 'h) -> (permissible (administer_treatment 'h 'p))].
let daxiom6 : [('p : patient) -> (critical_condition 'p) -> (obligatory (monitor_patient h2 'p))].
let daxiom7 : [('h : healthcare_professional) -> ('p : patient) -> (conscious 'p) -> (permissible (discharge_patient 'h 'p))].
let daxiom8 : [('h : healthcare_professional) -> (low_resources 'h) -> (impermissible (initiate_rehabilitation 'h 'p))].
let daxiom9 : [('h : healthcare_professional) -> ('p : patient) -> (assigned_care 'h 'p) -> (permissible (provide_palliative_care 'h 'p))].
let daxiom10 : [('h : healthcare_professional) -> ('e : equipment) -> (not_assigned_care 'h p1) -> (not (obligatory (use_equipment 'h p1 'e)))].
"""

theory_axioms = """let taxiom1 : [('p : patient) -> (critical_condition 'p) -> (red 'p)].
let taxiom2 : [('p : patient) -> (stable_vitals 'p) -> (green 'p)].
let taxiom3 : [('p : patient) -> (unconscious 'p) -> (severe 'p)].
let taxiom4 : [(immobile p2) -> (moderate p2)].
let taxiom5 : [('h : healthcare_professional) -> (high_resources 'h) -> (general_practitioner 'h)].
let taxiom6 : [('p : patient) -> (critical_condition 'p) -> (unstable_vitals 'p)].
let taxiom7 : [(conscious p2) -> (stable_vitals p2)].
let taxiom8 : [('h : healthcare_professional) -> (low_resources 'h) -> (nurse 'h)].
let taxiom9 : [('p : patient) -> (assigned_care h1 'p) -> (stable_condition 'p)].
let taxiom10 : [('h : healthcare_professional) -> ('p : patient) -> (not_assigned_care 'h 'p) -> (critical_condition 'p)].
"""


domain_text = "Create a task for an AI assistant managing the medical triage decisions."