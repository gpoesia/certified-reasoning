system_context = """You are an AI assistant, help the user with different tasks for deontic logic.
Here is the theory for the domain.
{theory}
"""
example_context = "{context}"

def get_context_prompt(system_context, example_context):
    messages_context = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": "Generate a context."},
        {"role": "assistant", "content": example_context},
        {"role": "user", "content": "Generate another context, varibales dont carry over. This should be different from the previous one. Remeber to use let in statements always."},
    ]
    return messages_context


axiom_templates = "Deoontic Axioms:\n{deontic_axioms}\nTheory Axioms:\n{theory_axioms}"
example_axioms = "Deoontic Axioms:\n{deontic_axioms}\nTheory Axioms:\n{theory_axioms}"
system_axioms = """You are an AI assistant, help the user with different tasks for deontic logic.
Here is the theory for the domain.
{theory}. You are asked to generate axioms based on templates. Be sure to stick to the templates.
Use 'type for generics (example 'e : event) and directly the variable names for others (example: e1 and not 'e1). Be sure to always declare generics before using them (like ('e: event) -> (free a1 'e)).
Dont write contradictory axioms.
"""

def get_axiom_prompt(system_axioms, axiom_templates, example_axioms, example_context, context):
    messages_axioms = [
        {"role": "system", "content": system_axioms},
        {"role": "user", "content": f"Generate axioms using these templates: {axiom_templates} \nUsing the following context: {example_context}\n Generate 10 deontic based axioms and 10 theory axioms."},
        {"role": "assistant", "content": example_axioms},
        {"role": "user", "content": f"Generate axioms using these templates: {axiom_templates} \nUsing the following context: {context}\n Generate 10 deontic based axioms and 10 theory axioms. Use different props in axiom, dont use similar props in the same axiom such as long, short. Relate some of the axioms to each other so that one can be used after the other."},
    ]
    return messages_axioms
