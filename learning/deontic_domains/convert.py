import copy
import os
import re

import openai

PROMPT_TEMPLATE = [
    {'role': 'system', 'content': 'You are a helpful assistant that helps the user rewrite their files into the correct, canonical format, demonstrated by your first response.'},
    {'role': 'user', 'content': '''Problem #1
Scenario:
In a non-profit organization, there are three volunteers: Alice, Bob, and Carol. They have three upcoming events: a fundraising event, a training event, and a charity gala. Alice is the organizer of the fundraising event, and Carol is a participant. The fundraising event is public. The training event is a short event. The fundraising event is a conference, and the charity gala is a social event. The training event occurs monthly. Alice gets an individual invite to the fundraising event.
If a person is free during an event, it is obligatory for them to check their availability for that event.
If a person is a participant in an event, it is permissible to delegate the event to them.
If a person is the organizer of an event, it is obligatory to update the event to be a social event.
For long events, it is permissible to set a reminder for a few days before the event for them.
If a person is busy during an event, it is impermissible to reschedule the event daily.
If a person is free during an event, it is obligatory for them to suggest an alternative time for that event.
If a person is the organizer of an event, it is impermissible for them to suggest an alternative time for that event.
If an event is long, it is permissible to change its visibility to confidential.
If an event is social, it must be public.
If a person is busy during an event, their priority for that event is low.
Public events are long events.
If an event is long, Carol is free for that event.
If a person is free during an event, their priority for that event is low.
Short events are conferences.
Question:
Given the situation and the rules of the world, is it obligatory for Carol to suggest an alternative time for the charity gala?
Scenario:
In a non-profit organization, there are three volunteers: Alice, Bob, and Carol. The organization also has a local team called "CommunityHelpers". They have three upcoming events: a fundraising event, a training event, and a charity gala. Alice is the organizer of the fundraising event, and Carol is a participant. The fundraising event is public. The training event is a short event. The fundraising event is a conference, and the charity gala is a social event. The training event occurs monthly. Alice gets an individual invite to the fundraising event.
For this organization, the following rules apply:
If a person is free during an event, it is obligatory for them to check their availability for that event.
If a person is a participant in an event, it is permissible to delegate the event to them.
If an event is social, it must be public.
If a person is free during an event, it is obligatory for them to suggest an alternative time for that event.
If a person is the organizer of an event, it is impermissible for them to suggest an alternative time for that event.
If a person is the organizer of an event, it is obligatory to update the event to be a social event.
If an event is long, it is permissible to change its visibility to confidential.
Public events are long events.
For long events, it is permissible to set a reminder for a few days before the event for them.
If an event is long, Carol is free for that event.
If a person is free during an event, their priority for that event is low.
Question: Given the situation and the rules of the world, is it obligatory for Carol to suggest an alternative time for the charity gala?
Solution:
To determine whether it is obligatory for Carol (b3) to suggest an alternative time for the charity gala (f3), we need to follow the reasoning trace:
(taxiom1 f3 cat3) : (public f3) – This states that the charity gala (f3) is a public event since it is a social event.
(taxiom3 f3 r1) : (long f3) – This states that the charity gala (f3) is a long event since it is public.
(taxiom4 f3 r2) : (free b3 f3) – This states that Carol (b3) is free during the charity gala (f3) since it is a long event.
(daxiom19 f3 b3 r3) : (obligatory (suggest_alternative_time b3 f3)) – This states that it is obligatory for Carol (b3) to suggest an alternative time for the charity gala (f3) since she is free during the event.
Answer:
Yes, it is obligatory for Carol (b3) to suggest an alternative time for the charity gala (f3).'''},
    {'role': 'assistant', 'content': '''Problem #1
Context: 1- In a non-profit organization, there are three volunteers: Alice, Bob, and Carol. 2- They have three upcoming events: a fundraising event, a training event, and a charity gala. 3- Alice is the organizer of the fundraising event, and Carol is a participant. 4- The fundraising event is a conference, and the charity gala is a social event. 5- The training event occurs monthly. 6- Alice gets an individual invite to the fundraising event. 7- If a person is free during an event, it is obligatory for them to check their availability for that event. 8- If a person is a participant in an event, it is permissible to delegate the event to them. 9- If an event is social, it must be public. 10- If a person is free during an event, it is obligatory for them to suggest an alternative time for that event. 11- If a person is the organizer of an event, it is impermissible for them to suggest an alternative time for that event. 12- If an event is long, it is permissible to change its visibility to confidential. 13- Public events are long events. 14- If an event is long, Carol is free for that event. 15- If a person is free during an event, their priority for that event is low.
Question: Given the rules above, is it obligatory for Carol to suggest an alternative time for the charity gala?
Reasoning: The charity gala is a public event. The charity gala is a long event. Carol is free during the charity gala. It is obligatory for Carol to suggest an alternative time for the charity gala.
Answer (Yes or no): Yes, it is obligatory for Carol to suggest an alternative time for the charity gala.'''},
]


def convert(path):
    print('Converting', path)
    with open(path) as input_file:
        content = input_file.read().strip()

    prompt = copy.deepcopy(PROMPT_TEMPLATE)
    prompt.append({
        'role': 'user',
        'content': f'Problem#2\n{content}'
    })

    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=prompt, max_tokens=1000)
    result = response['choices'][0]['message']['content']

    # Discard the 'Problem #2' line
    result = '\n'.join(result.split('\n')[1:])

    with open(os.path.basename(path), 'w') as f:
        f.write(result)
        print('Wrote', f.name)


def renumber(path):
    print('Renumbering', path)
    with open(path) as input_file:
        content = input_file.read().strip()

    lines = list(filter(None, map(lambda s: s.strip(), content.split('\n'))))

    for i, l in enumerate(lines):
        a, b = l.split(': ', 1)

        if b.startswith('1- '):
            sentence_regex = r'\b\d+\-\s+'
            sentences = list(map(lambda s: s.strip(), re.split(sentence_regex, b)[1:]))

            renumbered = []

            for j, s in enumerate(sentences):
                renumbered.append(f'{j+1}- {s}')

            new_l = a + ': ' + ' '.join(renumbered)

            if new_l != lines[i]:
                print('Before:', l)
                print('After:', new_l)

                c = input('Correct? (y/n)')

                if c != 'y':
                    raise ValueError('Oops')
                lines[i] = new_l

    result = '\n'.join(lines)

    with open(os.path.basename(path), 'w') as f:
        f.write(result)
        print('Wrote', f.name)


def main():
    for p in os.listdir('orig'):
        # convert(f'orig/{p}')
        renumber(p)


if __name__ == '__main__':
    main()
