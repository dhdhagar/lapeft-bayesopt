from lapeft_bayesopt.problems.prompting import PromptBuilder


class MyPromptBuilder(PromptBuilder):
    def __init__(self, kind: str, hint: str = ""):
        self.kind = kind
        self.hint = hint

    def get_prompt(self, x: str, obj_str: str) -> str:
        if self.kind == 'completion':
            return f'The estimated {obj_str} of the molecule {x} is: '
        elif self.kind == 'just-smiles':
            return x
        elif self.kind == 'word':
            # TwentyQuestions
            return x
        elif self.kind == 'instruction':
            # TwentyQuestions
            return f'The task is to find a hidden test word by guessing new words. What is a word that is similar to {x}?'
        elif self.kind == 'hint':
            # TwentyQuestions
            return f'The task is to find a hidden test word by guessing new words.{self.hint} Our next guess is {x}.'
        else:
            return NotImplementedError
