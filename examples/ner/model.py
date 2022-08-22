from typing import List

import torch
import torch.nn as nn
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from .metrics.span_to_label_f1 import SpanToLabelF1
from .modules.feature_extractor import NERFeatureExtractor
from transformers import  LukeForEntitySpanClassification


@Model.register("span_ner")
class ExhaustiveNERModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        feature_extractor: NERFeatureExtractor,
        dropout: float = 0.1,
        label_name_space: str = "labels",
        text_field_key: str = "tokens",
        prediction_save_path: str = None,
    ):
        super().__init__(vocab=vocab)
        
        self.feature_extractor = feature_extractor
        self.teacher_model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

        if torch.cuda.is_available():
            print("## TEACHER TO CUDA")
            self.teacher_model.cuda(device=1)
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        self.text_field_key = text_field_key

        self.biLstm = nn.LSTM(self.feature_extractor.get_output_dim(),
                            300,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        
        self.classifier = nn.Linear(600, vocab.get_vocab_size(label_name_space))

        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.span_f1 = SpanToLabelF1(self.vocab, prediction_save_path=prediction_save_path)
        self.span_accuracy = CategoricalAccuracy()

    def forward(
        self,
        word_ids: TextFieldTensors,
        entity_start_positions: torch.LongTensor,
        entity_end_positions: torch.LongTensor,
        original_entity_spans: torch.LongTensor,
        doc_id: List[str],
        labels: torch.LongTensor = None,
        entity_ids: torch.LongTensor = None,
        entity_position_ids: torch.LongTensor = None,
        input_words: List[List[str]] = None,
        **kwargs,
    ):
        feature_vector = self.feature_extractor(
            word_ids[self.text_field_key], 
            entity_start_positions,
            entity_end_positions, 
            entity_ids, 
            entity_position_ids
        )

        teacher_input = {} #= word_ids[self.text_field_key]
        #print(f"KEYS:: {word_ids[self.text_field_key].keys()}")
        teacher_input['input_ids'] = word_ids[self.text_field_key]['token_ids']
        teacher_input['entity_start_positions'] = entity_start_positions
        teacher_input['entity_end_positions'] = entity_end_positions
        teacher_input['attention_mask']=word_ids[self.text_field_key]['mask']
        teacher_input["entity_ids"] = entity_ids
        teacher_input["entity_position_ids"] = entity_position_ids
        teacher_input["entity_attention_mask"] = entity_ids != 0


        teacher_outputs = self.teacher_model(**teacher_input)

        #print(f"##### TEACHER LOGITS: {teacher_outputs.logits.shape}")
        feature_vector = self.dropout(feature_vector)
        lstm_out, (ht, ct) = self.biLstm(feature_vector)
        #!print(f"\n\n### LSTM OUT {lstm_out.shape}\n\n")
        feature_vector = self.dropout2(lstm_out)
        logits = self.classifier(feature_vector)

        prediction_logits, prediction = logits.max(dim=-1)
        output_dict = {"logits": logits, "prediction": prediction, "input": input_words}

        #if labels is not None:

            #print(f"## LABELS in Train: {labels.shape}")
            #print(f"## LOGITS in Train: {logits.shape}")

            ## TODO [SALEM] REPLACE LABELS WITH LOGITS HERE
        

        

        flattened_teacher_logits = teacher_outputs.logits.flatten(0,1)
        #print(f"### TEACHER LOGITS {flattened_teacher_logits.shape}")
        #print(f"#### FLATTENED LABELS BEFORE{labels.flatten().shape}")

        flattened_labels = labels.flatten().unsqueeze(1).repeat(1, flattened_teacher_logits.shape[1])
        #print(f"#### FLATTENED LABELS {flattened_labels.shape}")
        teacher_label_balanced = (0.5* flattened_teacher_logits) + (0.5*flattened_labels)
        flattened_student_logits = logits.flatten(0, 1)
        logprob = nn.functional.log_softmax(flattened_student_logits)

        loss = nn.functional.kl_div(logprob, teacher_label_balanced, reduction='batchmean')
        #print(f"### LABELS LOGITS {labels.flatten().shape}")
        #!print(f"### TEACHER LOGITS {flattened_teacher_logits.shape}")
        #!print(f"### STUDENT LOGITS {flattened_student_logits.shape}")

        output_dict["loss"] =   loss#self.criterion(flattened_student_logits, flattened_teacher_logits)
        self.span_accuracy(logits, labels, mask=(labels != -1))
        self.span_f1(prediction, labels, prediction_logits, original_entity_spans, doc_id, input_words)

        return output_dict

    def get_metrics(self, reset: bool = False):
        output_dict = self.span_f1.get_metric(reset)
        output_dict["span_accuracy"] = self.span_accuracy.get_metric(reset)
        return output_dict
