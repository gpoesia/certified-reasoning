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


let daxiom11 : [('h : healthcare_professional) -> ('p : patient) -> (critical_condition 'p) -> (obligatory (perform_examination 'h 'p))].
let daxiom12 : [('h : healthcare_professional) -> ('p : patient) -> (stable_condition 'p) -> (obligatory (assign_green 'h 'p))].
let daxiom13 : [('h : healthcare_professional) -> ('p : patient) -> (infected 'p) -> (impermissible (discharge_patient 'h 'p))].
let daxiom14 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (initiate_rehabilitation 'h 'p))].
let daxiom15 : [('h : healthcare_professional) -> ('p : patient) -> (low_resources 'h) -> (not (obligatory (administer_treatment 'h 'p)))].
let daxiom8 : [('h : healthcare_professional) -> ('p : patient) -> (high_resources 'h) -> (obligatory (discharge_patient 'h 'p))].
let daxiom16 : [('h : healthcare_professional) -> ('p : patient) -> (mobile 'p) -> (permissible (create_treatment_plan 'h 'p))].
let daxiom17 : [('h : healthcare_professional) -> ('e : equipment) -> (not_assigned_care 'h p3) -> (not (obligatory (use_equipment 'h p3 'e)))].
let daxiom18 : [('h : healthcare_professional) -> ('p : patient) -> (conscious 'p) -> (not (obligatory (discharge_patient 'h 'p)))].
let daxiom19 : [('h : healthcare_professional) -> ('p : patient) -> (unconscious 'p) -> (obligatory (monitor_patient 'h 'p))].
let daxiom20 : [('h : healthcare_professional) -> ('p : patient) -> (assigned_care 'h 'p) -> (obligatory (provide_family_support 'h 'p))].
let taxiom1 : [('p : patient) -> (infected 'p) -> (unstable_vitals 'p)].
let taxiom2 : [('p : patient) -> (pregnant 'p) -> (unstable_vitals 'p)].
let taxiom3 : [('p : patient) -> (stable_condition 'p) -> (conscious 'p)].
let taxiom4 : [('h : healthcare_professional) -> ('p : patient) -> (conscious 'p) -> (not_assigned_care 'h 'p)].
let taxiom5 : [('h : healthcare_professional) -> ('p : patient) -> (high_resources 'h) -> (assigned_care 'h 'p)].
let taxiom6 : [('h : healthcare_professional) -> ('p : patient) -> (declare_emergency 'h) -> (critical_condition 'p)].
let taxiom7 : [('h : healthcare_professional) -> ('p : patient) -> (assess_mental_health 'h 'p) -> (unstable_mental_state 'p)].
let taxiom8 : [('h : healthcare_professional) -> ('p : patient) -> (provide_family_support 'h 'p) -> (stable_mental_state 'p)].
let taxiom9 : [('p : patient) -> (assign_orange h4 'p) -> (critical_condition 'p)].
let taxiom10 : [('p : patient) -> (unstable_vitals 'p) -> (assign_orange h4 'p)].

Result:
(taxiom3 p3 stab2) : (conscious p3)
(daxiom18 h5 p3 r1) : (not (obligatory (discharge_patient h5 p3)))