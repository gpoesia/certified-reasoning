import os

def parse_and_negate_problem(file_path):
    with open(file_path, 'r') as f:
        problem_text = f.readlines()
    print(problem_text)
    
    # Extract the context
    context = problem_text[0].strip()

    # Extract the question
    question = problem_text[1].strip()

    # Replace terms with their negations
    negation_dict = {
        'permissible': 'impermissible',
        'impermissible': 'permissible',
        'obligatory': 'not obligatory',
        'not obligatory': 'obligatory'
    }

    if 'impermissible' in question:
        question = question.replace('impermissible', 'permissible')
    elif 'permissible' in question:
        question = question.replace('permissible', 'impermissible')
    elif 'not obligatory' in question:
        question = question.replace('not obligatory', 'obligatory')
    elif 'obligatory' in question:
        question = question.replace('obligatory', 'not obligatory')

    answer = "Answer (Yes or no): No"
    return context, question, answer

files = sorted([f for f in os.listdir('./') if f.endswith('.txt') and 'negated' not in f])
file_template = '{context}\n{question}\nReasoning:\n{answer}'
for f in files: 
    context, question, answer = parse_and_negate_problem(f)
    with open(f.replace('.txt', '_negated.txt'), 'w') as f:
        f.write(file_template.format(context=context, question=question, answer=answer))
    # print('Context:', context)
    # print('Question:', question)
    # print('Answer:', answer)
