# axioms for deontic logic
import random

deontic_templates = [
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} ({prop} {entity1} '{dtype1[0]})))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ('{dtype2[0]} : {dtype2}) -> ({deontic} ({action} ({prop} '{dtype2[0]} '{dtype1[0]})))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ('{dtype2[0]} : {dtype2}) -> ({deontic} ({action} '{dtype2[0]} '{dtype1[0]}))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} '{dtype1[0]}))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} ({prop} '{dtype1[0]})))].",
"let daxiom{i} : [({deontic} ({action} ({prop} ({entity1} {entity2})))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ('{dtype2[0]} : {dtype2}) -> ({deontic} ({action} '{dtype1[0]})) -> ({deontic} ({action} '{dtype2[0]}))]."
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} {entity1} '{dtype1[0]}))].",
"let daxiom{i} : [({deontic} ({action} {entity1})) -> ({deontic} ({action2} {entity2}))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} {entity1})) -> ({deontic} ({action2} '{dtype1[0]}))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ({deontic} ({action} ({prop1} {entity1} '{dtype1[0]})))].",
"let daxiom{i} : [('{dtype1[0]} : {dtype1}) -> ('{dtype2[0]} : {dtype2}) -> ({deontic} ({action} ({prop1} '{dtype2[0]} '{dtype1[0]})))].",
]

# axioms for extending the theory
theory_templates = [
    "let taxiom{i} : [('{dtype1[0]} : {dtype1})) -> ({prop1} {entity1} '{dtype1[0]})]",
    "let taxiom{i} : [('{dtype1[0]} : {dtype1})) -> ({prop1} {entity1} '{dtype1[0]}) -> ({prop2} {entity1} '{dtype1[0]})]",
    "let taxiom{i} : [('{dtype1[0]} : {dtype1}) -> ('{dtype2[0]} : {dtype2})) -> ({prop1} {entity1} '{dtype1[0]}) -> ({prop2} {entity1} '{dtype2[0]})]",
    "let taxiom{i} : [({prop1} {entity1} '{dtype1[0]}) -> ({prop2} {entity1} '{dtype1[0]})]",
    "let taxiom{i} : [({prop1} {entity1}) -> ({prop2} {entity1})]",
    "let taxiom{i} : [('{dtype1[0]} : {dtype1})) -> ({prop1} {entity1} '{dtype1[0]})]",
]
# theory_axiom_template_1 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b})]."
# theory_axiom_template_2 = "let axiom{i} : [({a}) -> ('{b[0]} : {b})]."
# # example = let axiom1 : [('e : event) -> ((individual_invite alice 'e): invite) -> (short 'e)]
# # theory_axiom_template_3 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> ('{c[0]} : {c})]."
# theory_axiom_template_3 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> ('{c[0]} : {c})]."
# theory_axiom_template_4 = "let axiom{i} : [('{a[0]} : {a}) -> ({b})]."

action_dict = {
    "send_notification": {"num_args": 1, "arg_types": ["invite"]},
    "accept": {"num_args": 1, "arg_types": ["invite"]},
    "decline": {"num_args": 1, "arg_types": ["invite"]},
    "cancel_event": {"num_args": 1, "arg_types": ["event"]},
    "set_reminder": {"num_args": 1, "arg_types": ["reminder"]},
    "add_participant": {"num_args": 2, "arg_types": ["event", "entity"]},
    "remove_participant": {"num_args": 2, "arg_types": ["event", "entity"]},
    "delegate_event": {"num_args": 2, "arg_types": ["event", "person"]},
    "update_event": {"num_args": 2, "arg_types": ["event", "event -> prop"]},
    "reschedule_event": {"num_args": 2, "arg_types": ["event", "event -> prop"]},
    "request_event_update": {"num_args": 2, "arg_types": ["person", "event"]},
    "suggest_alternative_time": {"num_args": 2, "arg_types": ["person", "event"]},
    "check_availability": {"num_args": 2, "arg_types": ["person", "event"]},
    "change_visibility": {"num_args": 2, "arg_types": ["event", "event -> prop"]},
}

# def sample_template_and_fill():

#     actions = ["send_notification", "accept", "decline", "cancel_event", "set_reminder", "add_participant", "remove_participant", "delegate_event", "update_event", "reschedule_event", "request_event_update", "suggest_alternative_time"]
#     types = ["entity", "person", "group", "event", "invite"]
#     props = ["short", "long", "high", "low", "daily", "weekly", "monthly", "yearly", "busy", "free", "tentative", "meeting", "conference", "social", "personal", "public", "private", "confidential"]
#     deontics = ["obligatory", "permissible", "impermissible"]
#     # 

#     chosen_template = random.choice(templates)
#     chosen_action = random.choice(actions)
#     chosen_entity = random.choice(entities)
#     chosen_type = random.choice(types)
#     chosen_prop = random.choice(props)
#     chosen_deontic = random.choice(deontics)

#     filled_template = chosen_template.format(i=random.randint(1, 100), action=chosen_action, entity=chosen_entity, type=chosen_type, prop=chosen_prop, deontic=chosen_deontic, dtype1=chosen_type, dtype1_0=chosen_type[0], dtype2=chosen_type, dtype2_0=chosen_type[0], entity1=chosen_entity, entity2=chosen_entity, action2=chosen_action)

#     return filled_template
