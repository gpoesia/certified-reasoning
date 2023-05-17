import os

def parse_and_negate_problem(file_path):
    with open(file_path, 'r') as f:
        problem_text = f.readlines()
    print(problem_text)
    
    # Extract the context
    context = problem_text[0].strip()

    # Extract the question
    question = problem_text[1].strip()

    if 'impermissible' in question:
        question_1 = question.replace('impermissible', 'permissible')
        question_2 = question.replace('impermissible', 'obligatory')
    elif 'permissible' in question:
        question_1 = question.replace('permissible', 'impermissible')
        question_2 = question.replace('permissible', 'obligatory')
    elif 'not obligatory' in question:
        question_1 = question.replace('not obligatory', 'obligatory')
        question_2 = question.replace('not obligatory', 'impermissible')
    elif 'obligatory' in question:
        question_1 = question.replace('obligatory', 'not obligatory')
        question_2 = question.replace('obligatory', 'impermissible')

    answer = "Answer (Yes or no): No"
    return context, [question_1, question_2], answer

files = sorted([f for f in os.listdir('./') if f.endswith('.txt') and 'negated' not in f and 'triage' in f])
file_template = '{context}\n{question}\nReasoning:\n{answer}'
for f in files: 
    context, questions, answer = parse_and_negate_problem(f)
    for i in range(2):
        with open(f.replace('.txt', f'_negated_{i}.txt'), 'w') as pf:
            pf.write(file_template.format(context=context, question=questions[i], answer=answer))

    # print('Context:', context)
    # print('Question:', question)
    # print('Answer:', answer)
