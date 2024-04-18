from transformers import LlamaModel, LlamaConfig
from .base import BaseLLMRegressor
from .utils import LLMFeatureType


class Llama2Regressor(BaseLLMRegressor):

    def __init__(self, kind, tokenizer, reduction=LLMFeatureType.AVERAGE, n_hidden_units=100, n_outputs=1):
        assert kind in ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']

        kind = f'meta-llama/{kind.capitalize()}-hf'
        config = LlamaConfig.from_pretrained(kind)
        config.attn_dropout = 0
        feature_extractor = LlamaModel.from_pretrained(kind, config=config)

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
