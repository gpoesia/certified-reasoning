let h4 : healthcare_professional.
let h5 : healthcare_professional.
let p3 : patient.
let p4 : patient.
let eq3 : equipment.
let eq4 : equipment.

let t3 : (physical_therapy p3).
let t4 : (medication p4).

let res3 : (low_resources h4).
let stab2 : (stable_condition p3).

let cons2 : (unconscious p3).
let mob3 : (mobile p4).

let inf1 : (infected p3).
let preg1 : (pregnant p4).


let daxiom3 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (discharge_patient 'h 'p))].
let daxiom11 : [('h : healthcare_professional) -> ('p : patient) -> (critical_condition 'p) -> (obligatory (perform_examination 'h 'p))].
let daxiom12 : [('h : healthcare_professional) -> ('p : patient) -> (stable_condition 'p) -> (obligatory (assign_green 'h 'p))].
let daxiom13 : [('h : healthcare_professional) -> ('p : patient) -> (infected 'p) -> (impermissible (discharge_patient 'h 'p))].
let daxiom14 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (initiate_rehabilitation 'h 'p))].
let daxiom15 : [('h : healthcare_professional) -> ('p : patient) -> (low_resources 'h) -> (not (obligatory (administer_treatment 'h 'p)))].
let daxiom17 : [('h : healthcare_professional) -> ('e : equipment) -> (not_assigned_care 'h p3) -> (not (obligatory (use_equipment 'h p3 'e)))].
let taxiom1 : [('p : patient) -> (infected 'p) -> (unstable_vitals 'p)].
let taxiom3 : [('p : patient) -> (stable_condition 'p) -> (conscious 'p)].
let taxiom5 : [('h : healthcare_professional) -> ('p : patient) -> (high_resources 'h) -> (assigned_care 'h 'p)].
let taxiom7 : [('h : healthcare_professional) -> ('p : patient) -> (assess_mental_health 'h 'p) -> (unstable_mental_state 'p)].
let taxiom9 : [('p : patient) -> (assign_orange h4 'p) -> (critical_condition 'p)].
let taxiom10 : [('p : patient) -> (unstable_vitals 'p) -> (assign_orange h4 'p)].

Result:
(daxiom13 h4 p3 inf1) : (impermissible (discharge_patient h4 p3))