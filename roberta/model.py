from transformers import RobertaModel, RobertaPreTrainedModel
from torch import nn


class RobertaExtraction(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaExtraction, self).__init__(config)

        self.config = config
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        self.start_output = nn.Linear(self.roberta.config.hidden_size, 1)
        self.intermediate_dense = nn.Linear(self.roberta.config.hidden_size * 2, 1)
        self.end_output = nn.Linear(self.roberta.config.hidden_size, 1)
        self.answerable_output = nn.Linear(self.roberta.config.hidden_size, 2)
        self.activation = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights()

    def forward(self, input_ids, attention_mask):

        roberta_output = self.roberta(input_ids, attention_mask).last_hidden_state
        roberta_output = self.dropout(roberta_output)

        answerable_logits = self.answerable_output(roberta_output[:, 0, :])

        start_logits = self.start_output(roberta_output)

        end_logits = self.end_output(roberta_output)

        return start_logits, end_logits, answerable_logits

    def evaluate(self, input_ids, attention_mask):

        roberta_output = self.roberta(input_ids, attention_mask).last_hidden_state

        start_logits = self.start_output(roberta_output)
        end_logits = self.end_output(roberta_output)
        answerable_logits = self.answerable_output(roberta_output[:, 0, :])

        start_logits = self.activation(start_logits)
        end_logits = self.activation(end_logits)
        answerable_logits = self.activation(answerable_logits)

        return start_logits, end_logits, answerable_logits
