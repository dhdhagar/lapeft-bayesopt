from lapeft_bayesopt.problems.prompting import PromptBuilder


class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str, hint: str):
        self.kind = kind
        self.hint = hint

    def get_prompt(self, x: str, obj_str: str, additive: bool = False) -> str:
        # Sober Look experiments:
        # if self.kind == 'completion':
        #     return f'The estimated {obj_str} of the molecule {x} is: '
        # elif self.kind == 'just-smiles':
        #     return x

        # TwentyQuestions
        instruction = 'The task is to find a hidden test word by guessing new words.'
        next_word = f'The next guess is {x}.'
        goodness = f'Is that a good guess?'
        hint = self.hint  # e.g. for "computer": "Hint: the hidden word is an example of a machine."

        if self.kind == 'word':
            return [x]
        elif self.kind == 'instruction':
            if not additive:
                return [f'{instruction} {next_word}']
            else:
                return [instruction, next_word]
        elif self.kind == 'hint':
            assert self.hint is not None and self.hint != ""
            if not additive:
                return [f'{instruction} {hint} {next_word}']
            else:
                return [instruction, hint, next_word]
        elif self.kind == 'hint-goodness':
            assert self.hint is not None and self.hint != ""
            if not additive:
                return [f'{instruction} {hint} {next_word} {goodness}']
            else:
                return [instruction, hint, next_word, goodness]
        else:
            return NotImplementedError
