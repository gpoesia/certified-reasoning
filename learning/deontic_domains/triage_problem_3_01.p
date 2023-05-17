let h4 : healthcare_professional.
let h5 : healthcare_professional.
let h6 : healthcare_professional.
let p3 : patient.
let p4 : patient.
let eq3 : equipment.
let eq4 : equipment.

let t3 : (physical_therapy p3).
let t4 : (medication p4).

let stab2 : (stable_condition p3).

let inf1 : (infected p3).
let preg1 : (pregnant p4).

let cons2 : consent = (consent_given p3 t3).
let cons3 : consent = (consent_withdrawn p4 t4).

let care2 : (assigned_care h5 p3).
let care3 : (assigned_care h6 p4).

let mental1 : (assess_mental_health h4 p3).
let mental2 : (provide_mental_health_support h5 p4).

let daxiom11 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (perform_examination 'h 'p))].
let daxiom12 : [('h : healthcare_professional) -> ('p : patient) -> (infected 'p) -> (obligatory (administer_treatment 'h 'p))].
let daxiom19 : [('h : healthcare_professional) -> ('p : patient) -> (infected 'p) -> (impermissible (provide_family_support 'h 'p))].
let daxiom14 : [('h : healthcare_professional) -> (consent_given p3 t3) -> (obligatory (administer_treatment 'h p3 t3))].
let daxiom15 : [('h : healthcare_professional) -> (consent_withdrawn p4 t4) -> (impermissible (administer_treatment 'h p4 t4))].
let daxiom16 : [('h : healthcare_professional) -> ('p : patient) -> (yellow 'p) -> (obligatory (initiate_life_support 'h 'p))].
let daxiom17 : [('h : healthcare_professional) -> ('p : patient) -> (stable_condition 'p) -> (not (obligatory (allocate_staff 'h 'p)))].
let daxiom18 : [('h : healthcare_professional) -> ('p : patient) -> (pregnant 'p) -> (permissible (provide_family_support 'h 'p))].
let daxiom1 : [('h : healthcare_professional) -> ('p : patient) -> (orange 'p) -> (not (obligatory (initiate_life_support 'h 'p)))].
let daxiom20 : [('h : healthcare_professional) -> ('e : equipment) -> (assigned_care 'h p3) -> (permissible (use_equipment 'h p3 'e))].
let taxiom1 : [('p : patient) -> (pregnant 'p) -> (not_infected 'p)].
let taxiom2 : [('h : healthcare_professional) -> ('p : patient) -> (assess_mental_health 'h 'p) -> (stable_mental_state 'p)].
let taxiom3 : [('p : patient) -> (stable_condition 'p) -> (conscious 'p)].
let taxiom5 : [('p : patient) -> (not_infected 'p) -> (yellow 'p)].
let taxiom6 : [('h : healthcare_professional) -> (low_resources 'h) -> (low_staff 'h)].
let taxiom10 : [('p : patient) -> (yellow 'p) -> (stable_condition 'p)].
let taxiom4 : [('p : patient) -> ('t : treatment) -> (mobile 'p) -> (consent_given 'p)].
let taxiom9 : [('h : healthcare_professional) -> ('p : patient) -> (mobile 'p) -> (initiate_rehabilitation 'h 'p)].

Result:
(taxiom1 p4 preg1) : (not_infected p4)
(taxiom5 p4 r1) : (yellow p4)
(daxiom16 h6 p4 r2) : (obligatory (initiate_life_support h6 p4))