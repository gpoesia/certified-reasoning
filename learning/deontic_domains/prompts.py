# Prompts for generating the context
system_context = """You are an AI assistant, help the user with different tasks for deontic logic.
Here is the theory for the domain.
{theory}
Each context structure should be different.
"""
example_context = "{context}"

def get_context_prompt(system_context, example_context):
    messages_context = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": "Generate a context."},
        {"role": "assistant", "content": example_context},
        {"role": "user", "content": "Generate another context, variables dont carry over. This should be different from the previous one. Remeber to use let in statements always. Be creative."},
    ]
    return messages_context


# Prompts for generating the axioms
calendar_bad_taxioms = """
let taxiom3 : [('f : event) -> (recurrence 'f) -> (daily 'f) -> (weekly 'f)].
let taxiom4 : [('f : event) -> (priority 'f) -> (high b2 'f) -> (low b1 'f)].
"""
calendar_good_taxioms = """

"""
axiom_templates = "Deoontic Axioms:\n{deontic_axioms}\nTheory Axioms:\n{theory_axioms}"
example_axioms = "Deoontic Axioms:\n{deontic_axioms}\nTheory Axioms:\n{theory_axioms}"
system_axioms = """You are an AI assistant, help the user with different tasks for deontic logic.
Here is the theory for the domain.
{theory}. You are asked to generate axioms based on templates. Be sure to stick to the templates.
Use 'type for generics (example 'e : event) and directly the variable names for others (example: e1 and not 'e1, b1 and not 'b1). Be sure to always declare generics before using them (like ('e: event) -> (free a1 'e)).
Dont write contradictory axioms.
For theory aximos or taxioms, use different types of props within an axiom. Relate the axioms to each other so that one can follow from the other. Be especially creative with taxioms.
These are bad axioms:
let taxiom : [('f : event) -> (recurrence 'f) -> (daily 'f) -> (weekly 'f)]. ; as daily and weekly are similar (both are recurrence)
let taxiom : [('f : event) -> (priority 'f) -> (high b2 'f) -> (low b1 'f)]. ; as high and low are similar (both are priority)
These are good axioms:
let taxiom : [('e : event) -> (yearly 'e) -> (long 'e)]. ; as yearly (recurrence) and long (duration) are different
let taxiom : [('e: event) -> (meeting 'e) -> (private 'e)]. ; as meeting (category) and private (visibility) are different
let taxiom : [('p : person) -> ('e : event) -> (high 'e 'p) -> (free 'e 'p)]. ; as high (priority) and free (availability) are different
"""

def get_axiom_prompt(system_axioms, axiom_templates, example_axioms, example_context, context):
    messages_axioms = [
        {"role": "system", "content": system_axioms},
        {"role": "user", "content": f"Generate axioms using these templates: {axiom_templates} \nUsing the following context: {example_context}\n Generate 10 deontic based axioms and 10 theory axioms."},
        {"role": "assistant", "content": example_axioms},
        {"role": "user", "content": f"Generate axioms using these templates: {axiom_templates} \nUsing the following context: {context}\n Generate 10 deontic based axioms and 10 good theory axioms."},
    ]
    return messages_axioms

# prompts for generating the text
system_text = """Generate different tasks for deontic logic. Make the tasks realistic. Be sure to stick to the templates. {domain_text}
Here is the theory for the domain.
{theory}."""



def get_text_prompt(domain_text, theory, example_context, example_axioms, example_answer, example_story, context, axioms, answer):
    messages = [
        {"role": "system", "content": system_text.format(domain_text=domain_text, theory=theory)},
        {"role": "user", "content": f"For a problem, this is the context:\n{example_context}\nHere are the rules of the world in this scenario:\n{example_axioms}\nHere is the reasoning trace for the problem:{example_answer}"},
        {"role": "assistant", "content": example_story},
        {"role": "user", "content": f"For the next problem, this is the context:\n{context}\nHere are the rules of the world in this scenario:\n{axioms}\nHere is the reasoning trace for the problem:{answer}\n Make the context (the characters and story) different from the previous one. Be sure to include all the axioms in the rules."},
    ]
    return messages
