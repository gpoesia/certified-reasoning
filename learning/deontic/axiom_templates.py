# axioms for deontic logic
deontic_axiom_template_1 = "let axiom{i} : [('{a[0]} : {a}) -> (permissible ({action} ({t} {entity} '{a[0]})))]."
deontic_axiom_template_2 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> (permissible ({action} ({t} '{b[0]} '{a[0]})))]."
deontic_axiom_template_3 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> (permissible ({action} '{b[0]} '{a[0]}))]."
deontic_axiom_template_4 = "let axiom{i} : [('{a[0]} : {a}) -> (permissible ({action} '{a[0]}))]."
deontic_axiom_template_5 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> ('{c[0]} : {c}) -> (permissible ({action} '{a[0]} '{b[0]} '{c[0]}))]."

# axioms for extending the theory
theory_axiom_template_1 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b})]."
theory_axiom_template_2 = "let axiom{i} : [({a}) -> ('{b[0]} : {b})]."
# example = let axiom1 : [('e : event) -> ((individual_invite alice 'e): invite) -> (short 'e)]
# theory_axiom_template_3 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> ('{c[0]} : {c})]."
theory_axiom_template_3 = "let axiom{i} : [('{a[0]} : {a}) -> ('{b[0]} : {b}) -> ('{c[0]} : {c})]."
theory_axiom_template_4 = "let axiom{i} : [('{a[0]} : {a}) -> ({b})]."

def populate_axiom_template(template, values):
    populated_template = template.format(**values)
    return populated_template
