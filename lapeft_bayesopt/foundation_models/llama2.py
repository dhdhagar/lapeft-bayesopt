from transformers import LlamaModel, LlamaConfig, AutoModelForCausalLM
from .base import BaseLLMRegressor


class LLMFeatureType(Enum):
    LAST_TOKEN = 1
    FIRST_TOKEN = 2
    AVERAGE = 3


class Llama2Regressor(BaseLLMRegressor):

    def __init__(self, kind, tokenizer, reduction=LLMFeatureType.AVERAGE, n_hidden_units=100, n_outputs=1,
                 dtype=None, vtoken=False):
        assert kind in ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']

        self.kind = f'meta-llama/{kind.capitalize()}-hf'
        add_args = {}
        config = LlamaConfig.from_pretrained(self.kind)
        config.attn_dropout = 0
        if dtype is not None:
            add_args['torch_dtype'] = dtype
        if vtoken:
            feature_extractor = AutoModelForCausalLM.from_pretrained(self.kind, config=config, **add_args)
        else:
            feature_extractor = LlamaModel.from_pretrained(self.kind, config=config, **add_args)

        super().__init__(
            tokenizer=tokenizer,
            reduction=reduction,
            feature_dim=feature_extractor.config.hidden_size,
            n_hidden_units=n_hidden_units,
            n_outputs=n_outputs
        )

        self.config = config
        self.feature_extractor = feature_extractor

    def forward_features(self, data):
        device = next(self.parameters()).device
        input_ids = data['input_ids'].to(device, non_blocking=True)
        # Adding missing attention mask
        attention_mask = data['attention_mask'].to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids, attention_mask=attention_mask).last_hidden_state
        return feat
