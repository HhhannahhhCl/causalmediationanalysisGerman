import inspect
import os
import re

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from experiment import Model, Intervention

# Stats from https://arxiv.org/pdf/1804.06876.pdf, Table 1
OCCUPATION_FEMALE_PCT = {
    'carpenter': 2,
    'mechanic': 4,
    'construction worker': 4,
    'laborer': 4,
    'driver': 6,
    'sheriff': 14,
    'mover': 18,
    'developer': 20,
    'farmer': 22,
    'guard': 22,
    'chief': 27,
    'janitor': 34,
    'lawyer': 35,
    'cook': 38,
    'physician': 38,
    'ceo': 39,
    'analyst': 41,
    'manager': 43,
    'supervisor': 44,
    'salesperson': 48,
    'editor': 52,
    'designer': 54,
    'accountant': 61,
    'auditor': 61,
    'writer': 63,
    'baker': 65,
    'clerk': 72,
    'cashier': 73,
    'counselor': 73,
    'attendant': 76,
    'teacher': 78,
    'tailor': 80,
    'librarian': 84,
    'assistant': 85,
    'cleaner': 89,
    'housekeeper': 89,
    'nurse': 90,
    'receptionist': 90,
    'hairdresser': 92,
    'secretary': 95
}

'''NOT NEEDED! new version added below
def load_dev_examples(path='winobias_data/', verbose=False):
    return load_examples(path, 'dev', verbose)

def load_test_examples(path='winobias_data/', verbose=False):
    return load_examples(path, 'test', verbose)
'''

#def load_winoexamples(path='winobias_data/', verbose=False): # Change up in all scripts
#    return load_examples(path, 'dev', verbose)

def load_examples(path, verbose=False):
    with open('Data/female_occupations.txt', encoding='UTF-8') as f:#(os.path.join(path, 'Data/female_occupations.txt')) as f:
        female_occupations = [row.strip() for row in f] # removed .lower() in both cases
    with open('Data/male_occupations.txt', encoding='UTF-8') as f:
        male_occupations = [row.strip() for row in f]
    occupations = female_occupations + male_occupations
    #print(occupations)

    #fname = f'pro_stereotyped_type1.txt'

    with open(path, encoding='UTF-8') as f:
        examples = []
        row_pair = []
        skip_count = 0
        for row in f:
            row = re.sub('\n', '', row)
            row_pair.append(row)
            if len(row_pair) == 2:
                #print(row_pair)
                skip = False
                if row_pair[0].count('[') != 2 or row_pair[1].count('[') != 2: # Multiple pronouns
                    skip = True
                    print('first')
                else:
                    base_string1, substitutes1, continuation1, occupation1 = _parse_row(row_pair[0], occupations)
                    #print(base_string1)
                    base_string2, substitutes2, continuation2, occupation2 = _parse_row(row_pair[1], occupations)
                    if base_string1 != base_string2 or substitutes1 != substitutes2:
                        skip = True
                        print('second')
                if skip:
                    if verbose:
                        print('Skipping: ', row_pair)
                    skip_count += 1
                    row_pair = []
                    continue
                base_string = base_string1
                assert substitutes1 == substitutes2
                female_pronoun, male_pronoun = substitutes1
                assert len(continuation1) > 0 and len(continuation2) > 0 and continuation1 != continuation2
                assert len(occupation1) > 0 and len(occupation2) > 0 and occupation1 != occupation2
                #print(base_string, base_string1, base_string2, occupation1, occupation2, substitutes1, substitutes2, continuation1, continuation2)
                if occupation1 in female_occupations:
                    female_occupation = occupation1
                    female_occupation_continuation = continuation1
                    male_occupation = occupation2
                    male_occupation_continuation = continuation2
                    assert occupation2 in male_occupations
                else:
                    male_occupation = occupation1
                    male_occupation_continuation = continuation1
                    female_occupation = occupation2
                    female_occupation_continuation = continuation2
                    assert occupation1 in male_occupations
                    assert occupation2 in female_occupations
                examples.append(WinobiasExample(base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation))
                row_pair = [] # emptied again
        assert row_pair == []
    print(f'Loaded {len(examples)} pairs. Skipped {skip_count} pairs.')
    return examples


def _parse_row(row, occupations): # Maybe do this yourself as the data set was created based on these conditions
    sentence = row.strip() # _, sentence = row.strip().split(' ', 1)
    occupation = None
    for occ in occupations:
        if f'[{occ}' in sentence: # .lower() removed for both, also removed second bracket for different declinations
            assert occupation is None
            occupation = occ # .lower()
    assert occupation is not None

    pronoun_groups = [ # First element is female, second is male
        ('sie', 'er')#, # nominative
        #('her', 'his') # possessive
    ]

    num_matches = 0
    substitutes = None
    for pronouns in pronoun_groups: 
        pattern = '|'.join(r'\[' + p + r'\]' for p in pronouns) # matches '[he]', '[she]', etc.
        pronoun_matches = re.findall(pattern, sentence) # Find all pronouns in one sentence
        assert len(pronoun_matches) <= 1 # check True or False
        if pronoun_matches:
            num_matches += 1
            pronoun_match = pronoun_matches[0] # If pronoun in sentence, take first one and split sentence after
            # print(sentence)
            context, continuation = sentence.split(pronoun_match)
            # print(context, continuation)
            context = context.replace('[', '').replace(']', '') # replace all brackets in prompt and strip
            context = context.strip()
            # print(context)
            assert '[' not in continuation  # Check for brackets in continuation (should not be the case)
            continuation = continuation.strip()
            substitutes = pronouns
    assert num_matches == 1 # check whether there was actually a match
    base_string = context + ' {}'

    return base_string, substitutes, continuation, occupation


def _odds_ratio(model, female_context, male_context, candidates):
    prob_female_occupation_continuation_given_female_pronoun, prob_male_occupation_continuation_given_female_pronoun = \
        model.get_probabilities_for_examples_multitoken(female_context, candidates)
    prob_female_occupation_continuation_given_male_pronoun, prob_male_occupation_continuation_given_male_pronoun = \
        model.get_probabilities_for_examples_multitoken(male_context, candidates)

    odds_given_female_pronoun = prob_female_occupation_continuation_given_female_pronoun / \
                                prob_male_occupation_continuation_given_female_pronoun
    odds_given_male_pronoun = prob_female_occupation_continuation_given_male_pronoun / \
                              prob_male_occupation_continuation_given_male_pronoun
    return odds_given_female_pronoun / odds_given_male_pronoun


def analyze(examples, gpt2_version='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(gpt2_version)

    model = Model(gpt2_version=gpt2_version)
    data = []
    for ex in tqdm(examples):
        candidates = [ex.female_occupation_continuation, ex.male_occupation_continuation]
        substitutes = [ex.female_pronoun, ex.male_pronoun]
        intervention = Intervention(tokenizer, ex.base_string, substitutes, candidates)
        female_context = intervention.base_strings_tok[0]
        male_context = intervention.base_strings_tok[1]
        odds_ratio = _odds_ratio(model, female_context, male_context, intervention.candidates_tok)
        female_pronoun = female_context[-1:]
        male_pronoun = male_context[-1:]
        # be aware that this uses the list, i.e. not correct German (Zimmer/Psycholog) or generics masculine
        odds_ratio_no_context = _odds_ratio(model, female_pronoun, male_pronoun, intervention.candidates_tok)
        desc = f'{ex.base_string.replace("{}", ex.female_pronoun + "/" + ex.male_pronoun)} // {ex.female_occupation_continuation} // {ex.male_occupation_continuation}'
        # Check these two lines --> essential?
        #female_occupation_female_pct = OCCUPATION_FEMALE_PCT[ex.female_occupation]
        #male_occupation_female_pct = OCCUPATION_FEMALE_PCT[ex.male_occupation]

        data.append({'odds_ratio': odds_ratio,
                     'odds_ratio_no_context': odds_ratio_no_context,
                     'female_occupation': ex.female_occupation,
                     'male_occupation': ex.male_occupation,
                     'desc': desc# , 'occupation_pct_ratio': female_occupation_female_pct / male_occupation_female_pct
                     })
    return pd.DataFrame(data)


class WinobiasExample():

    def __init__(self, base_string, female_pronoun, male_pronoun, female_occupation, male_occupation,
                 female_occupation_continuation, male_occupation_continuation):
        self.base_string = base_string
        self.female_pronoun = female_pronoun
        self.male_pronoun = male_pronoun
        self.female_occupation = female_occupation
        self.male_occupation = male_occupation
        self.female_occupation_continuation = female_occupation_continuation
        self.male_occupation_continuation = male_occupation_continuation

    def to_intervention(self, tokenizer):
        return Intervention(
            tokenizer=tokenizer,
            base_string=self.base_string,
            substitutes=[self.female_pronoun, self.male_pronoun],
            candidates=[self.female_occupation_continuation, self.male_occupation_continuation]
        )

    def __str__(self):
        return inspect.cleandoc(f"""
            base_string: {self.base_string}
            female_pronoun: {self.female_pronoun}
            male_pronoun: {self.male_pronoun}
            female_occupation: {self.female_occupation}
            male_occupation: {self.male_occupation}
            female_occupation_continuation: {self.female_occupation_continuation}
            male_occupation_continuation: {self.male_occupation_continuation}
        """)

    def __repr__(self):
        return str(self).replace('\n', ' ')


if __name__ == "__main__":
    load_test_examples(verbose=True)
