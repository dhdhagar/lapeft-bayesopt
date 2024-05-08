import torch
from transformers import LlamaTokenizer, T5Tokenizer
from lapeft_bayesopt.foundation_models.t5 import T5Regressor
from lapeft_bayesopt.foundation_models.llama2 import Llama2Regressor
from enum import Enum


class LLMFeatureType(Enum):
    LAST_TOKEN = 1
    FIRST_TOKEN = 2
    AVERAGE = 3


def average_llm_features(feats, input_ids, pad_token_id):
    # Masking---0 for everything after <eos> and 1 otherwise
    mask = ~input_ids.eq(pad_token_id).to(feats.device)
    mask = mask[:, :, None].float()  # (batch_size, seq_len, 1)
    #  (batch_size, 1, hidden_size) / (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
    feats = (feats * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
    return feats


def extract_last_llm_features(feats, input_ids, eos_token_id):
    # Find the last token position (before padding)
    eos_mask = input_ids.eq(eos_token_id).to(feats.device)
    if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
        raise ValueError('All examples must have the same number of <eos> tokens.')
    batch_size, _, hidden_size = feats.shape
    feats = feats[eos_mask, :]  # (batch_size, hidden_size)
    feats = feats.view(batch_size, -1, hidden_size)[:, -1, :]
    return feats


def get_llama2_tokenizer(kind):
    kind = f'meta-llama/{kind.capitalize()}-hf'
    tokenizer = LlamaTokenizer.from_pretrained(kind)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_t5_tokenizer(kind):
    tokenizer = T5Tokenizer.from_pretrained(kind)  # , model_max_length=512)
    return tokenizer


def get_tokenizer_and_model(kind, dtype, is_vtoken=False):
    if kind.startswith('t5'):
        tokenizer = get_t5_tokenizer(kind)
        llm_feat_extractor = T5Regressor(
            kind=kind,
            tokenizer=tokenizer,
            encoder_only=not is_vtoken,
            dtype=dtype
        )
    elif kind.startswith('llama'):
        tokenizer = get_llama2_tokenizer(kind)
        llm_feat_extractor = Llama2Regressor(
            kind=kind,
            tokenizer=tokenizer,
            dtype=dtype,
            vtoken=is_vtoken
        )
    return tokenizer, llm_feat_extractor
