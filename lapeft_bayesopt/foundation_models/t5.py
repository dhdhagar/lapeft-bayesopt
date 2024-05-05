from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Config
from .base import BaseLLMRegressor
from .utils import LLMFeatureType


class T5Regressor(BaseLLMRegressor):

    def __init__(self, kind, tokenizer, reduction=LLMFeatureType.AVERAGE, n_hidden_units=100, n_outputs=1,
                 encoder_only=True, dtype=None):
        assert kind in ['t5-small', 't5-base', 't5-large', 'GT4SD/multitask-text-and-chemistry-t5-base-augm']

        self.kind = kind
        add_args = {}
        config = T5Config.from_pretrained(self.kind)
        config.dropout_rate = 0
        if dtype is not None:
            add_args['torch_dtype'] = dtype
        if encoder_only:
            feature_extractor = T5EncoderModel.from_pretrained(self.kind, config=config, **add_args)
        else:
            feature_extractor = T5ForConditionalGeneration.from_pretrained(self.kind, config=config, **add_args)

        super().__init__(
            tokenizer=tokenizer,
            reduction=reduction,
            feature_dim=feature_extractor.config.d_model,
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
