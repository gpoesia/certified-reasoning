action : type.
not : [prop -> prop].
/* base deontic types */
obligatory : [action -> prop].
permissible : [action -> prop].
impermissible : [action -> prop].

/* base axioms */
ob_perm : [(obligatory 'a) -> (permissible 'a)].
im_perm : [(impermissible 'a) -> (not (permissible 'a))].

/* entity types */
entity : type.
healthcare_professional : entity.
patient : entity.

/* priority types */
priority : type.
red : [patient -> prop].
orange : [patient -> prop].
yellow : [patient -> prop].
green : [patient -> prop].

/* Assign Priority Levels */
assign_red : [healthcare_professional -> patient -> action].
assign_orange : [healthcare_professional -> patient -> action].
assign_yellow : [healthcare_professional -> patient -> action].
assign_green : [healthcare_professional -> patient -> action].

injury_disease : type.
minor : [patient -> prop].
moderate : [patient -> prop].
severe : [patient -> prop].
critical : [patient -> prop].

treatment : type.
medication : [patient -> treatment].
surgery : [patient -> treatment].
physical_therapy : [patient -> treatment].

symptom : type.
check_symptoms : [patient -> action].

examination : type.
perform_examination : [healthcare_professional -> patient -> action].

diagnosis : type.
diagnose : [healthcare_professional -> patient -> action].

treatment_plan : type.
create_treatment_plan : [healthcare_professional -> patient -> action].
administer_treatment : [healthcare_professional -> patient -> action].

monitor : type.
monitor_patient : [healthcare_professional -> patient -> action].

discharge : type.
discharge_patient : [healthcare_professional -> patient -> action].

resource_availability : type.
high_resources : [healthcare_professional -> prop].
low_resources : [healthcare_professional -> prop].
check_resources : [healthcare_professional -> action].

staff_availability : type.
high_staff : [healthcare_professional -> prop].
low_staff : [healthcare_professional -> prop].
check_staff : [healthcare_professional -> action].

/* Vital Signs */
vital_signs : type.
stable_vitals : [patient -> prop].
unstable_vitals : [patient -> prop].

/* Medical Expertise */
medical_expertise : type.
general_practitioner : [healthcare_professional -> prop].
specialist : [healthcare_professional -> prop].
nurse : [healthcare_professional -> prop].
paramedic : [healthcare_professional -> prop].

/* Patient Condition */
patient_condition : type.
stable_condition : [patient -> prop].
critical_condition : [patient -> prop].

/* Allergies */
allergies : type.
has_allergies : [patient -> prop].
no_allergies : [patient -> prop].

/* Medical History */
medical_history : type.
has_medical_history : [patient -> prop].
no_medical_history : [patient -> prop].

/* Consciousness Level */
consciousness_level : type.
conscious : [patient -> prop].
unconscious : [patient -> prop].

/* Mobility */
mobility : type.
mobile : [patient -> prop].
immobile : [patient -> prop].

/* Infection */
infection : type.
infected : [patient -> prop].
not_infected : [patient -> prop].

/* Pregnancy */
pregnancy : type.
pregnant : [patient -> prop].
not_pregnant : [patient -> prop].

/* Mental State */
mental_state : type.
stable_mental_state : [patient -> prop].
unstable_mental_state : [patient -> prop].

/* Patient consent */
consent : type.
provide_consent : [patient -> treatment -> action].
withdraw_consent : [patient -> treatment -> action].

/* Emergency */
emergency : type.
declare_emergency : [healthcare_professional -> action].
end_emergency : [healthcare_professional -> action].

/* Patient History */
patient_history : type.
check_patient_history : [healthcare_professional -> patient -> action].

/* Medical Equipment */
equipment : type.
use_equipment : [healthcare_professional -> patient -> equipment -> action].
check_equipment_availability : [healthcare_professional -> action].


/* Resource Allocation */
resource_allocation : type.
allocate_resources : [healthcare_professional -> patient -> action].

/* Staff Allocation */
staff_allocation : type.
allocate_staff : [healthcare_professional -> patient -> action].

/* Hydration and Nutrition */
hydration_nutrition : type.
provide_hydration_nutrition : [healthcare_professional -> patient -> action].

/* Rehabilitation */
rehabilitation : type.
initiate_rehabilitation : [healthcare_professional -> patient -> action].

/* Palliative Care */
palliative_care : type.
provide_palliative_care : [healthcare_professional -> patient -> action].

/* Mental Health */
mental_health : type.
assess_mental_health : [healthcare_professional -> patient -> action].
provide_mental_health_support : [healthcare_professional -> patient -> action].

/* Family Support */
family_support : type.
provide_family_support : [healthcare_professional -> patient -> action].

/* Life Support */
life_support : type.
initiate_life_support : [healthcare_professional -> patient -> action].
end_life_support : [healthcare_professional -> patient -> action].

/* Care Responsibility */
care_responsibility : type.
assigned_care : [healthcare_professional -> patient -> prop].
not_assigned_care : [healthcare_professional -> patient -> prop].

/* Follow-up Plan */
follow_up_plan : type.
has_follow_up_plan : [healthcare_professional -> patient -> prop].
no_follow_up_plan : [healthcare_professional -> patient -> prop].

/* Confidentiality Agreement */
confidentiality_agreement : type.
confidentiality_agreed : [healthcare_professional -> patient -> prop].
confidentiality_not_agreed : [healthcare_professional -> patient -> prop].

