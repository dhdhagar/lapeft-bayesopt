from transformers import T5ForConditionalGeneration, T5Config, AutoModelForCausalLM, \
    AutoTokenizer, T5Tokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, \
    DataCollatorWithPadding, get_scheduler, EarlyStoppingCallback, IntervalStrategy
from transformers.trainer_callback import ProgressCallback
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
import schedulefree
import torch
import datasets
import pandas as pd
import os
import random


def get_tokenized_dataset(data, tokenizer, _type="seq2seq"):
    if type(data['prompts']) is not list or len(data['prompts']) == 1:
        # If data has only one prompt element, we will use it as the target and use a dummy input as the prompt
        dummy_prompt = True
        raw_dataset = [
            {'prompt': '#', 'target': data['prompts'] if type(data['prompts']) is str else data['prompts'][0]}]
    else:
        # If data has more than one prompt element, we will use the last prompt as the target
        # Note: this means it's not compatible with prompt_strategy=hint-goodness
        dummy_prompt = False
        raw_dataset = [{'prompt': ' '.join(data['prompts'][:-1]), 'target': data['prompts'][-1]}]
    hf_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=raw_dataset))
    hf_dataset = hf_dataset.map(lambda x: tokenizer(" ".join(x["prompt"]), text_target=x["target"]))
    need_eos = hf_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id
    if _type != "seq2seq":
        # "causal"
        hf_dataset = hf_dataset.map(lambda x: {
            **x,
            "input_ids": x["input_ids"] + x["labels"] + ([tokenizer.eos_token_id] if need_eos else []),
            "attention_mask": [0 if dummy_prompt else 1] * len(x["attention_mask"]) + [1] * len(x["labels"]) + (
                [1] if need_eos else []),
            "labels": [-100] * len(x["input_ids"]) + x["labels"] + ([tokenizer.eos_token_id] if need_eos else [])
        })
    else:
        # "seq2seq"
        hf_dataset = hf_dataset.map(lambda x: {
            **x,
            "input_ids": x["input_ids"] + ([tokenizer.eos_token_id] if need_eos else []),
            "attention_mask": [0 if dummy_prompt else 1] * (len(x["attention_mask"]) + (1 if need_eos else 0)),
            "labels": x["labels"] + ([tokenizer.eos_token_id] if need_eos else [])
        })
    return hf_dataset


def create_training_arguments(out_dir, learning_rate, epochs, device='cuda'):
    os.makedirs(os.path.join(out_dir, 'temp'), exist_ok=True)
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, 'temp'),
        use_cpu=device == 'cpu',
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        # logging_steps=epochs // 10,
        logging_strategy="no",  # disable logging
        eval_steps=epochs // 50,
        metric_for_best_model='accuracy',
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


class CustomProgressCallback(ProgressCallback):
    def __init__(self):
        super()
    def on_train_begin(self, args, state, control, **kwargs):
        pass
    def on_step_end(self, args, state, control, **kwargs):
        pass
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        pass
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
    def on_train_end(self, args, state, control, **kwargs):
        pass


def create_trainer(model, tokenizer, training_args, dataset, schedule_free=False, _type="seq2seq"):
    add_args = {}
    if schedule_free:
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=training_args.learning_rate,
            warmup_steps=training_args.num_train_epochs // 10,
        )
        add_args["optimizers"] = (optimizer, None)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) if _type == "seq2seq" else DataCollatorWithPadding(
        tokenizer)  # DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def compute_metrics(eval_pred):
        _type = "seq2seq" if type(eval_pred.predictions) is tuple else "causal"
        preds = eval_pred.predictions[0] if _type == "seq2seq" else eval_pred.predictions
        preds = preds.argmax(axis=-1).squeeze()  # greedy
        labels = eval_pred.label_ids.squeeze()
        is_correct = labels[labels != -100] == preds[:len(preds) if _type == "seq2seq" else (len(preds) - 1)][
            labels != -100]
        return {'accuracy': (is_correct.sum() / len(is_correct)) == 1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        callbacks=[CustomEarlyStoppingCallback()],  # , CustomProgressCallback()],
        compute_metrics=compute_metrics,
        **add_args
    )
    return trainer


def get_virtual_token(feature_extractor, tokenizer, data, out_dir, num_virtual_tokens=1,
                      learning_rate=40, epochs=1000, schedule_free=True, device='cuda'):
    model_name = feature_extractor.kind
    hf_dataset = get_tokenized_dataset(data, tokenizer, _type="seq2seq" if model_name.startswith('t5') else "causal")

    # Load peft model
    generation_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM if model_name.startswith('t5') else TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,  # PromptTuningInit.RANDOM
        prompt_tuning_init_text=tokenizer.decode(data['labels'][data['labels'].count(-100):], skip_special_tokens=True),
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=model_name,  # pre-trained model name
        num_transformer_submodules=1  # Force the vtoken to be added at the encoder only for encoder-decoder models
    )
    peft_model = get_peft_model(feature_extractor.feature_extractor, generation_config)
    # print(peft_model.print_trainable_parameters())

    # Get training args
    training_args = create_training_arguments(learning_rate=learning_rate * (4 if model_name.startswith('t5') else 1),
                                              epochs=epochs, out_dir=out_dir, device=device)
    # Get trainer
    trainer = create_trainer(model=peft_model, tokenizer=tokenizer, training_args=training_args, dataset=hf_dataset,
                             schedule_free=schedule_free, _type="seq2seq" if model_name.startswith('t5') else "causal")
    # Run training
    trainer.train()

    # Verify that the final validation accuracy is True
    assert trainer.state.log_history[-2]['eval_accuracy']

    # Return virtual token
    with torch.no_grad():
        virtual_token = peft_model.get_prompt(1)
    return virtual_token


def get_outputs(model, inputs=None, inputs_embeds=None, decoder_inputs_embeds=None, max_new_tokens=300, device='cuda',
                text=True):
    if inputs_embeds is not None or decoder_inputs_embeds is not None:
        outputs = model.generate(
            inputs_embeds=None if inputs_embeds is None else inputs_embeds.to(device),
            decoder_inputs_embeds=None if decoder_inputs_embeds is None else decoder_inputs_embeds.to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id
        )
    if text:
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs
