import json
import spacy
import torch
from torch import nn

from itertools import chain
from models.integrated_hierarchical_model import IntegratedHierarchicalModel
from typing import List, Dict

import data_process_utils as U


class TagPredictor:

    def __init__(self):
        self.model = None
        self.__load_model()

        self.tokenizer = spacy.load('en_core_web_sm')
        self.w2idx_dict = json.load(open('../res/word_to_index.json', 'r'))
        self.tag_list = list(json.load(open('../res/index_to_tag.json', 'r')).values())

        print('Initialized pipeline')

    def get_tag_list(self):
        return self.tag_list

    def get_prediction(self, narrative_input, review_input, limit=5, detail_result=False):
        """

        """
        feature_dict = self.__extract_features_for_one_instance(self.__tokenize(narrative_input),
                                                                self.__tokenize(review_input))
        prediction = self.model(torch.tensor(feature_dict['narrative_sequence_hierarchical']).unsqueeze(0),
                  torch.tensor(feature_dict['review_sequence_hierarchical']).unsqueeze(0))

        prediction = {k: v.detach().numpy() for k, v in prediction.items()}
        out_tags = prediction['out'][0].argsort()[::-1]
        out_tags = [self.tag_list[i] for i in out_tags]

        if detail_result:
            return out_tags[:limit], prediction
        else:
            return out_tags[:limit]


    def __tokenize(self, text, flatten=False):
        doc = self.tokenizer(text)
        sentences = []

        for sent in doc.sents:
            sentence = [t.text for t in sent if not t.text == '\n']
            sentences.append(sentence)

        if flatten:
            return list(chain.from_iterable(sentences))

        return sentences

    def __extract_features_for_one_instance(self, tokenized_narrative: List[List[str]],
                                            tokenized_review: List[List[str]]):
        """
        This function converts a tokenized synopsis and a tokenized review into a feature vector.
        The tokenized inputs are 2D lists. Each row is a sentence and each column is a token.

        """

        feature_dict = dict()

        feature_dict['tokenized_narrative'] = tokenized_narrative
        feature_dict['tokenized_review'] = tokenized_review

        narrative_lower = [[w.lower() for w in sent] for sent in tokenized_narrative]
        review_lower = [[w.lower() for w in sent] for sent in tokenized_review]

        # Narrative
        feature_dict['narrative_sequence_hierarchical'] = U.convert_text_to_hierarchical_sequence(narrative_lower,
                                                                                                  self.w2idx_dict,
                                                                                                  sentence_filter_len=6)
        feature_dict['narrative_sequence_hierarchical'] = U.pad_hierarchical_sequence(
            feature_dict['narrative_sequence_hierarchical'],
            len(feature_dict['narrative_sequence_hierarchical']),
            max(map(len, feature_dict['narrative_sequence_hierarchical']))
        )

        # Review
        try:
            feature_dict['review_sequence_hierarchical'] = U.convert_text_to_hierarchical_sequence(
                review_lower, self.w2idx_dict, sentence_filter_len=6)

            feature_dict['review_sequence_hierarchical'] = U.pad_hierarchical_sequence(
                feature_dict['review_sequence_hierarchical'],
                len(feature_dict['review_sequence_hierarchical']),
                max(map(len, feature_dict['review_sequence_hierarchical']))
            )
        except:
            feature_dict['review_sequence_hierarchical'] = [[0]]

        return feature_dict

    def __load_model(self):
        embedding_matrix = torch.Tensor(json.load(open('../res/pretrained_glove.json', 'r'))).float()

        self.model = IntegratedHierarchicalModel(embedding_matrix,
                                                 freeze_embedding=False,
                                                 sentence_encoder_dim=32,
                                                 document_encoder_dim=32,
                                                 pooling_method='attention',
                                                 encoder_type=nn.LSTM,
                                                 batch_size=32,
                                                 share_embedding=True,
                                                 feature_type='glove')

        checkpoint = torch.load('../res/best.pth', map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print('Model Loaded')

