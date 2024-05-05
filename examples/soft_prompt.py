from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Config, AutoModelForCausalLM, \
    AutoTokenizer, T5Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding, get_scheduler, \
    EarlyStoppingCallback, IntervalStrategy, TrainerCallback
from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
import schedulefree
import torch
import datasets
import pandas as pd
import os
import random
from IPython import embed

working_dir = "./"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_tokenized_dataset(tokenizer, shuffle_input=False, constant_input=None, seed=17):
    # Load dataset
    raw_dataset = [{'prompt': 'I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. \
I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type \
commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. \
my first command is pwd'},
                   {'prompt': 'I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. \
I want you to only reply with the terminal output inside one unique code block, and nothing else. do not write explanations. do not type \
commands unless I instruct you to do so. when i need to tell you something in english, i will do so by putting text inside curly brackets {like this}. \
my first command is chmod',
                    }]
    hf_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=raw_dataset))
    if not MODEL_NAME.startswith('t5'):
        hf_dataset = hf_dataset.map(
            lambda x: tokenizer(" ".join(x["prompt"].split()[:-1]), text_target=x["prompt"].split()[-1]))
        if shuffle_input:
            random.seed(seed)
            hf_dataset = hf_dataset.map(
                lambda x: {**x, "input_ids": random.sample(x["input_ids"], len(x["input_ids"]))})
        if constant_input is not None:
            hf_dataset = hf_dataset.map(lambda x: {**x, "input_ids": [constant_input] * len(x["input_ids"])})
        hf_dataset = hf_dataset.map(
            lambda x: {**x, "input_ids": x["input_ids"] + x["labels"] + [tokenizer.eos_token_id],
                       "attention_mask": x["attention_mask"] + [1] * len(x["labels"]) + [1],
                       "labels": [-100] * len(x["input_ids"]) + x["labels"] + [tokenizer.eos_token_id]})
        # hf_dataset = hf_dataset.map(lambda x: tokenizer(x["prompt"]))
        # hf_dataset = hf_dataset.map(lambda x: {**x, "input_ids": x["input_ids"]+[tokenizer.eos_token_id], "attention_mask": x["attention_mask"]+[1]})
    else:
        hf_dataset = hf_dataset.map(
            lambda x: tokenizer(" ".join(x["prompt"].split()[:-1]), text_target=x["prompt"].split()[-1]))
        # hf_dataset = hf_dataset.map(lambda x: tokenizer("Instruction", text_target=x["prompt"]))
        # hf_dataset = hf_dataset.map(lambda x: {**x, "attention_mask": [0]*len(x["attention_mask"])})  # Don't attend to the encoder input tokens (only attend to the encoder virtual token)
    return hf_dataset


def get_outputs(model, inputs=None, inputs_embeds=None, decoder_inputs_embeds=None, max_new_tokens=300, device='cuda',
                text=True):
    if inputs_embeds is not None or decoder_inputs_embeds is not None:
        outputs = model.generate(
            inputs_embeds=None if inputs_embeds is None else inputs_embeds.to(device),
            decoder_inputs_embeds=None if decoder_inputs_embeds is None else decoder_inputs_embeds.to(device),
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # do_sample=True,
            # repetition_penalty=1.5, #Avoid repetition.
            # early_stopping=True, #The model can stop before reach the max_length
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # do_sample=True,
            # repetition_penalty=1.5, #Avoid repetition.
            # early_stopping=True, #The model can stop before reach the max_length
            eos_token_id=tokenizer.eos_token_id
        )
    if text:
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs


def create_training_arguments(path, learning_rate=0.0035, epochs=6, device='cuda'):
    training_args = TrainingArguments(
        output_dir=path,  # Where the model predictions and checkpoints will be written
        use_cpu=device == 'cpu',  # This is necessary for CPU clusters.
        auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
        learning_rate=learning_rate,  # Higher learning rate than full Fine-Tuning
        num_train_epochs=epochs,
        logging_steps=epochs // 10,
        eval_steps=epochs // 10,
        metric_for_best_model='accuracy',  # 'loss',
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS
    )
    return training_args


class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self):
        super()

    def on_evaluate(self, args, state, control, **kwargs):
        is_correct = kwargs['metrics']['eval_accuracy']
        if is_correct:
            control.should_training_stop = True


def create_trainer(peft_model, training_args, train_dataset, schedule_free=False):
    add_args = {}
    if schedule_free:
        optimizer = schedulefree.AdamWScheduleFree(
            peft_model.parameters(),
            lr=training_args.learning_rate,
            warmup_steps=100,
        )
        add_args["optimizers"] = (optimizer, None)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model) if MODEL_NAME.startswith(
        't5') else DataCollatorWithPadding(tokenizer)  # DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def compute_metrics(eval_pred):
        _type = "seq2seq" if type(eval_pred.predictions) is tuple else "causal"
        preds = eval_pred.predictions[0] if _type == "seq2seq" else eval_pred.predictions
        preds = preds.argmax(axis=-1).squeeze()  # greedy
        labels = eval_pred.label_ids.squeeze()
        is_correct = labels[labels != -100] == preds[:len(preds) if _type == "seq2seq" else (len(preds) - 1)][
            labels != -100]
        return {'accuracy': (is_correct.sum() / len(is_correct)) == 1}

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[CustomEarlyStoppingCallback()],
        # [EarlyStoppingCallback(early_stopping_patience=1)],  # , early_stopping_threshold=0.2
        compute_metrics=compute_metrics,
        **add_args
    )
    return trainer


def load_and_set_adapter(directory, name):
    loaded_model.load_adapter(directory, adapter_name=name)
    loaded_model.set_adapter(name)
    return loaded_model


def get_virtual_token(foundational_model, tokenizer, hf_dataset, data_idx=0, num_virtual_tokens=1, learning_rate=3e-3,
                      epochs=500, save=True, reset=False, schedule_free=False):
    output_directory = os.path.join(working_dir, f"peft_model_{data_idx}")
    # Check if the model already exists
    if not reset and os.path.exists(output_directory):
        peft_model = PeftModel.from_pretrained(foundational_model,
                                               output_directory,
                                               device_map='auto',
                                               is_trainable=False)
    else:
        # Load peft model
        generation_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM if MODEL_NAME.startswith('t5') else TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            # PromptTuningInit.RANDOM if MODEL_NAME.startswith('t5') else PromptTuningInit.TEXT,  # PromptTuningInit.RANDOM,
            prompt_tuning_init_text=tokenizer.decode(
                hf_dataset[data_idx]['labels'][hf_dataset[data_idx]['labels'].count(-100):], skip_special_tokens=True),
            # hf_dataset[data_idx]['prompt'],  # only if using TEXT init
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=MODEL_NAME,  # pre-trained model name
            num_transformer_submodules=1
        )
        peft_model = get_peft_model(foundational_model, generation_config)
        print(peft_model.print_trainable_parameters())
        with torch.no_grad():
            init_vtoken = peft_model.get_prompt(1)

        # Create directories to store the models
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Get training args
        training_args = create_training_arguments(output_directory, learning_rate, epochs, device=device)
        # Get trainer
        trainer = create_trainer(peft_model, training_args,
                                 train_dataset=hf_dataset.select(range(data_idx, data_idx + 1)),
                                 schedule_free=schedule_free)
        # Run training
        trainer.train()
        # Get trained model
        peft_model = trainer.model
        # Save if required
        if save:
            peft_model.save_pretrained(output_directory)

    # Return virtual token
    with torch.no_grad():
        virtual_token = peft_model.get_prompt(1)

    return virtual_token, hf_dataset[data_idx]['prompt'], peft_model, trainer

global MODEL_NAME
MODEL_NAME = "t5-base" # "t5-base"  # "bigscience/bloomz-560m" # "bigscience/bloomz-560m"  # "gpt2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if MODEL_NAME in ['gpt2', 'meta-llama/Llama-2-7b-hf']:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset
hf_dataset = get_tokenized_dataset(tokenizer, shuffle_input=False)

# Load model
if MODEL_NAME.startswith('t5'):
    config = T5Config.from_pretrained(MODEL_NAME)
    config.dropout_rate = 0
    foundational_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=config, device_map='auto')
else:
    foundational_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if MODEL_NAME.startswith('meta') else torch.float32,
        device_map='auto'
    )

# Get vtoken
vtoken, prompt, peft_model, trainer = get_virtual_token(foundational_model, tokenizer, hf_dataset,
                                                        data_idx=1, num_virtual_tokens=1,
                                                        learning_rate=30*(4 if MODEL_NAME.startswith('t5') else 1),
                                                        epochs=100, schedule_free=True,
                                                        save=False, reset=True)

token_embeds = foundational_model.get_input_embeddings()
for p in token_embeds.parameters():
    break
# Based on prompt text
# _prompt = " ".join(prompt.split()[:-1])  # Everything except pwd/chmod
# prompt_embed = p[tokenizer(_prompt, return_tensors='pt')['input_ids'][0]][None, :, :]

# Based on prompt input_ids
if MODEL_NAME.startswith('t5'):
    prompt_input_ids = hf_dataset[0]['input_ids']  # encoder input
else:
    prompt_input_ids = hf_dataset[0]['input_ids'][:hf_dataset[0]['labels'].count(-100)]  # inputs for which no predictions need to be made
_prompt = tokenizer.decode(prompt_input_ids, skip_special_tokens=True)
prompt_embed = p[prompt_input_ids][None, :, :]

vtoken_plus_text = torch.cat((vtoken, prompt_embed), dim=1)  # prepend vtoken to prompt embed
vtoken_plus_text.shape

# Generate vtoken output and compare
vtoken_output = get_outputs(foundational_model, inputs_embeds=vtoken_plus_text.type(foundational_model.dtype),
                            device=device, text=True)[0]
print(f'Prompt:\n{_prompt}\n')
print(f'Vtoken:\n{vtoken_output}')
