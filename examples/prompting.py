from lapeft_bayesopt.problems.prompting import PromptBuilder


class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str, hint: str = ""):
        self.kind = kind
        self.hint = hint

    def get_prompt(self, x: str, obj_str: str, additive: bool = False) -> str:
        # if self.kind == 'completion':
        #     return f'The estimated {obj_str} of the molecule {x} is: '
        # elif self.kind == 'just-smiles':
        #     return x
        instruction = 'The task is to find a hidden test word by guessing new words.'
        next_word = f'The next guess is {x}.'
        hint = self.hint

        if self.kind == 'word':
            # TwentyQuestions
            return [x]
        elif self.kind == 'instruction':
            # TwentyQuestions
            # return f'The task is to find a hidden test word by guessing new words. What is a word that is similar to {x}?'
            if not additive:
                return [f'{instruction} {next_word}']
            else:
                return [instruction, _next_word]
        elif self.kind == 'hint':
            # TwentyQuestions
            if not additive:
                return [f'{instruction} {hint} {next_word}']
            else:
                return [instruction, hint, next_word]
        elif self.kind == 'hint-goodness':
            # TwentyQuestions
            if not additive:
                return [f'{instruction} {hint} {next_word} Is that a good guess?']
            else:
                return [instruction, hint, f'{next_word} Is that a good guess?']
        else:
            return NotImplementedError
