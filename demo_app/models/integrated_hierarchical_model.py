import torch
from torch import nn
from torch.nn import functional as F
from models.hierarchical_encoder import HierarchicalEncoder
from models.attention import Attention
import json
import parameters as P
import config as C

debug = False
bidirectional = True


def print_msg(*msg):
    if debug:
        print(msg)


class IntegratedHierarchicalModel(nn.Module):
    """
    The integrated model that encodes narratives and reviews and uses gating mechanism to combine them.


    """

    def __init__(self,
                 embedding_matrix,
                 freeze_embedding,
                 sentence_encoder_dim,
                 document_encoder_dim,
                 pooling_method,
                 encoder_type,
                 batch_size,
                 share_embedding=False,
                 feature_type='glove'):
        """

        Parameters
        ----------
        embedding_matrix: torch.FloatTensor
            Needed when Glove or Word2Vec static embeddings are used. Dimension is vocab_size*embedding_dimension

        freeze_embedding: Boolean
            Determines if the embedding matrix should be tuned or kept frozen

        sentence_encoder_dim: Integer
            Number of RNN cells to use in the sentence encoder

        document_encoder_dim: Integer
            Number of RNN cells to use in the document encoder

        pooling_method: String
             Pooling strategy in the encoder. Supported options are "attention" and "maxpool"

        encoder_type: torch.nn.Module
            Determined the RNN cell type. Supported options are torch.nn.LSTM and torch.nn.GRU

        batch_size: Integer
            Batch size

        share_embedding: Boolean
            For static embeddings, this parameter determines if encoders for narratives and reviews will share and tune
            the same embedding table or separate.

        feature_type: String
            Type of the input embedding features. Available options are "glove", "elmo", and "bert"
        """

        super(IntegratedHierarchicalModel, self).__init__()

        """ Preset values """
        self.linear_layer1_dim = 128
        self.linear_layer2_dim = 64
        self.output_dim = 71
        self.share_embedding = share_embedding

        self.sentence_encoder_dim = sentence_encoder_dim
        self.document_encoder_dim = document_encoder_dim

        """ Create the embedder. """
        if feature_type == 'glove':
            self.emb_dim = embedding_matrix.size()[-1]
            if self.share_embedding:
                self.embedder = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embedding)
            else:
                self.embedder = None
        elif feature_type == 'elmo' or feature_type == 'bert':
            self.embedder = None

        """ Encoder for the narratives """
        if P.NARRATIVE_ACTIVE:
            self.narrative_encoder = HierarchicalEncoder(embedding_matrix,
                                                         freeze_embedding=freeze_embedding,
                                                         sentence_encoder_dim=self.sentence_encoder_dim,
                                                         document_encoder_dim=self.document_encoder_dim,
                                                         pooling_method=pooling_method,
                                                         encoder_type=encoder_type,
                                                         batch_size=batch_size,
                                                         embedder=self.embedder,
                                                         feature_type=feature_type)

        """ Encoder for the reviews """
        if P.REVIEW_ACTIVE and not P.SHARED_ENCODER:
            self.review_encoder = HierarchicalEncoder(embedding_matrix,
                                                      freeze_embedding=freeze_embedding,
                                                      sentence_encoder_dim=self.sentence_encoder_dim,
                                                      document_encoder_dim=self.document_encoder_dim,
                                                      pooling_method=pooling_method,
                                                      encoder_type=encoder_type,
                                                      batch_size=batch_size,
                                                      embedder=self.embedder,
                                                      feature_type=feature_type)

        """ Module for combining the narratives and reviews """
        if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
            if P.GATED:
                self.representation_combiner = RepresentationCombiner2('gate', self.document_encoder_dim * 2)
            else:
                self.representation_combiner = RepresentationCombiner('simple_concat', self.document_encoder_dim * 2)

        """ Fully Connected layer """
        if bidirectional:
            if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
                if P.GATED:
                    fc_input_dim = self.document_encoder_dim * 2
                else:
                    fc_input_dim = self.document_encoder_dim * 4
            else:
                fc_input_dim = self.document_encoder_dim * 2
        else:
            fc_input_dim = self.document_encoder_dim
        print_msg('fc_input_dim 1', fc_input_dim)

        if P.USE_RESIDUAL:
            if P.NARRATIVE_ACTIVE and P.REVIEW_ACTIVE:
                fc_input_dim *= 3
            else:
                fc_input_dim *= 2

        print_msg('fc_input_dim 2', fc_input_dim)
        if P.SENT_LEVEL_PRED:
            fc_input_dim += self.output_dim
            self.sent_to_pred_linear = nn.Linear(self.sentence_encoder_dim * 2, self.output_dim)
            if P.SENT_LEVEL_POOL == 'attention':
                self.sent_to_pred_att = Attention(self.output_dim)

        if P.REVIEW_ACTIVE and P.SENT_LEVEL_PRED_REVIEW:
            fc_input_dim += self.output_dim
            self.sent_to_pred_linear_review = nn.Linear(self.sentence_encoder_dim * 2, self.output_dim)
            if P.SENT_LEVEL_POOL == 'attention':
                self.sent_to_pred_att_review = Attention(self.output_dim)
        print_msg('fc_input_dim 3', fc_input_dim)
        if P.SMALL_MLP:
            self.fc_out = nn.Sequential(
                nn.Dropout(P.DROPOUT_RATE),  # New addition in 8
                nn.Linear(fc_input_dim, self.output_dim)
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(fc_input_dim, self.linear_layer1_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.linear_layer1_dim),
                nn.Dropout(P.DROPOUT_RATE),
                nn.Linear(self.linear_layer1_dim, self.linear_layer2_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(self.linear_layer2_dim),
                nn.Linear(self.linear_layer2_dim, self.output_dim)
            )

        # self.class_weights = nn.Parameter(
        #    torch.Tensor(json.load(open(C.PROCESSED_DATA_PATH + '/class_weights.json', 'r'))), requires_grad=False)
        # """ Training utils"""
        # self.sent_batchnorm = nn.BatchNorm1d(self.sentence_encoder_dim * 2)
        # self.doc_batchnorm = nn.BatchNorm1d(self.document_encoder_dim * 2)

    def forward(self, narrative_features, review_features, narrative_mask=None, review_mask=None):
        """
        Steps:
        1. Encode the narrative and review using hierarchical encoder
        2. Use MLP module to predict final distribution

        Parameters
        ----------
        narrative_features: torch.LongTensor
            The document matrix of size B * M * N. Where,
                B: Batch size
                M: Longest document length in terms of number of sentences
                N: Longest sentence length in terms of number of words

        review_features: torch.LongTensor
        review_features: torch.LongTensor
        narrative_mask: torch.LongTensor
        review_mask: torch.LongTensor

        """
        combined_representation = None
        narrative_vector, review_vector = None, None
        review_sent_attention, review_doc_attention = None, None
        narrative_sent_attention, narrative_doc_attention = None, None
        sent_to_tag_weighted = None

        """ Run encoding on narrative and review """
        if P.NARRATIVE_ACTIVE:
            narrative_vector, narrative_doc_attention, narrative_sent_attention, narrative_encoded_sentences, \
            narrative_sentence_level_mask = self.narrative_encoder(narrative_features, narrative_mask)
            print_msg('N', narrative_vector.size(), narrative_sentence_level_mask.size())
        if P.REVIEW_ACTIVE:
            if not P.SHARED_ENCODER:
                review_vector, review_doc_attention, review_sent_attention, review_encoded_sentences, \
                review_sentence_level_mask = self.review_encoder(review_features, review_mask)
            else:
                review_vector, review_doc_attention, review_sent_attention, review_encoded_sentences, \
                review_sentence_level_mask = self.narrative_encoder(review_features, review_mask)
            print_msg('R', review_vector.size(), review_sentence_level_mask.size())

        """ [Conditional] Combine representations """
        if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
            gate_weights, combined_representation = self.representation_combiner([narrative_vector, review_vector])
        else:
            if P.NARRATIVE_ACTIVE:
                combined_representation = narrative_vector
            elif P.REVIEW_ACTIVE:
                combined_representation = review_vector
            gate_weights = None

        """ [Conditional] Residual connection """
        if P.GATED and P.USE_RESIDUAL:
            c_l = [combined_representation]
            if P.NARRATIVE_ACTIVE:
                c_l.append(narrative_vector)
            if P.REVIEW_ACTIVE:
                c_l.append(review_vector)
            combined_representation = torch.cat(c_l, -1)

        """ [Conditional] Predict tags for each encoded sentences """
        if P.SENT_LEVEL_PRED:
            print_msg('sent_to_tag_prob size', narrative_encoded_sentences.size(),
                      narrative_sentence_level_mask.unsqueeze(2).size())
            sent_to_tag_prob = self.sent_to_pred_linear(
                narrative_encoded_sentences) * narrative_sentence_level_mask.unsqueeze(2)

            # Earlier another attention was computed separately. Now using the doc attention
            # 32 15 71 X  32 15 (NEW)
            sent_to_tag_weighted = sent_to_tag_prob * narrative_doc_attention.unsqueeze(2)
            sent_to_tag_pooled = torch.sum(sent_to_tag_weighted, dim=1)

            # if P.SENT_LEVEL_POOL == 'max':
            #     sent_to_tag_pooled = F.max_pool1d(sent_to_tag_prob.permute(0, 2, 1), kernel_size=sent_to_tag_prob.size(1))
            #     sent_to_tag_pooled = sent_to_tag_pooled.reshape(sent_to_tag_pooled.size()[:2])
            # elif P.SENT_LEVEL_POOL == 'attention':
            #     sent_to_tag_pooled, sent_to_tag_att_weights, sent_to_tag_h = self.sent_to_pred_att(sent_to_tag_prob, narrative_sentence_level_mask)
            #     sent_to_tag_weighted = sent_to_tag_h * sent_to_tag_att_weights.unsqueeze(2)

            combined_representation = torch.cat([combined_representation, sent_to_tag_pooled], -1)

        if P.REVIEW_ACTIVE and P.SENT_LEVEL_PRED_REVIEW:
            sent_to_tag_prob_review = self.sent_to_pred_linear_review(
                review_encoded_sentences) * review_sentence_level_mask.unsqueeze(2)
            print_msg('sent_to_tag_prob size', sent_to_tag_prob_review.size())

            # Earlier another attention was computed separately. Now using the doc attention
            # 32 15 71 X  32 15 (NEW)
            sent_to_tag_weighted_review = sent_to_tag_prob_review * review_doc_attention.unsqueeze(2)
            sent_to_tag_pooled_review = torch.sum(sent_to_tag_weighted_review, dim=1)

            combined_representation = torch.cat([combined_representation, sent_to_tag_pooled_review], -1)

        """ Tag classification """
        output = self.fc_out(combined_representation)

        """ Final Activation """
        output = F.softmax(output, dim=1)
        # output = F.sigmoid(output)

        out_dict = {'out': output,
                    'plot_sent_attn': narrative_sent_attention,
                    'plot_doc_attn': narrative_doc_attention,
                    'review_body_sent_attn': review_sent_attention,
                    'review_body_doc_attn': review_doc_attention,
                    'gate_weights': gate_weights,
                    'review_title_sent_attn': torch.Tensor([]),
                    'review_title_doc_attn': torch.Tensor([]),
                    'sent_to_tag_weighted': sent_to_tag_weighted
                    }

        return out_dict


class RepresentationCombiner(nn.Module):

    def __init__(self, combine_method, incoming_data_size=32):
        """

        -----------
        combine_method: String
            simple_concat / gate
        incoming_data_size: Dict
            Keys: [plot_vector_size, review_body_size, review_title_size]
        gate_size: Integer
            Dimension of each gate
        """

        super(RepresentationCombiner, self).__init__()
        self.combine_method = combine_method
        self.incoming_data_size = float(incoming_data_size)

        if self.combine_method == 'gate':
            # plot gate
            self.narrative_w = nn.Linear(incoming_data_size, incoming_data_size)
            # self.rt_w = nn.Linear(incoming_data_size, incoming_data_size)
            self.review_w = nn.Linear(incoming_data_size, incoming_data_size)
            self.control_gate = nn.Linear(incoming_data_size * 2, incoming_data_size, bias=False)

            self.activation = nn.Tanh()  # or softsign

            self.softmax = nn.Softmax(dim=-1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, representation_list):

        if self.combine_method == 'simple_concat':
            null_gate_weights = torch.zeros(representation_list[0].size()[0], 2)
            return null_gate_weights, torch.cat(representation_list, dim=-1)

        elif self.combine_method == 'gate':
            # list: 0=plot, 1=review_body, 2=review_title
            h_plot = self.activation(self.narrative_w(representation_list[0]))
            # h_rt = self.activation(self.rt_w(representation_list[1]))
            h_rb = self.activation(self.review_w(representation_list[-1]))

            z = self.sigmoid(self.control_gate(torch.cat(representation_list, dim=-1)))

            print_msg('gate', h_plot.size(), h_rb.size(), z.size())
            # h = torch.cat([h_plot, h_rt, h_rb], dim=-1)
            h = z * h_plot + (1. - z) * h_rb

            plot_sig_activated = ((torch.sum(z > 0.5, dim=-1).float()) / self.incoming_data_size) * 100
            rb_sig_activated = ((self.incoming_data_size - torch.sum(z > 0.5,
                                                                     dim=-1)).float() / self.incoming_data_size) * 100
            gate_weights = torch.stack([plot_sig_activated, rb_sig_activated], dim=1)

            return gate_weights, h


class RepresentationCombiner2(nn.Module):

    def __init__(self, combine_method, incoming_data_size=32):
        """

        -----------
        combine_method: String
            simple_concat / gate
        incoming_data_size: Dict
            Keys: [plot_vector_size, review_body_size, review_title_size]
        gate_size: Integer
            Dimension of each gate
        """

        super(RepresentationCombiner2, self).__init__()
        self.combine_method = combine_method
        self.incoming_data_size = float(incoming_data_size)

        if self.combine_method == 'gate':
            # plot gate
            self.narrative_w = nn.Sequential(
                nn.Linear(incoming_data_size, 512),
                nn.ReLU(),
                nn.Linear(512, incoming_data_size)
            )
            # self.rt_w = nn.Linear(incoming_data_size, incoming_data_size)
            self.review_w = nn.Sequential(
                nn.Linear(incoming_data_size, 512),
                nn.ReLU(),
                nn.Linear(512, incoming_data_size)
            )
            self.control_gate = nn.Linear(incoming_data_size * 2, incoming_data_size, bias=False)

            self.activation = nn.Tanh()  # or softsign

            self.softmax = nn.Softmax(dim=-1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, representation_list):

        if self.combine_method == 'simple_concat':
            null_gate_weights = torch.zeros(representation_list[0].size()[0], 2)
            return null_gate_weights, torch.cat(representation_list, dim=-1)

        elif self.combine_method == 'gate':
            h_plot = self.activation(self.narrative_w(representation_list[0]))
            h_rb = self.activation(self.review_w(representation_list[-1]))

            z = self.sigmoid(self.control_gate(torch.cat(representation_list, dim=-1)))

            print_msg('gate', h_plot.size(), h_rb.size(), z.size())
            # h = torch.cat([h_plot, h_rt, h_rb], dim=-1)
            h = z * h_plot + (1. - z) * h_rb

            plot_sig_activated = ((torch.sum(z > 0.5, dim=-1).float()) / self.incoming_data_size) * 100
            rb_sig_activated = ((self.incoming_data_size - torch.sum(z > 0.5,
                                                                     dim=-1)).float() / self.incoming_data_size) * 100
            gate_weights = torch.stack([plot_sig_activated, rb_sig_activated], dim=1)

            return gate_weights, h


# Test
def test_model():
    import json

    class_weights = torch.FloatTensor(json.load(open(processed_data_dump_path + '/class_weights_1.json', 'r')))
    sentence_encoder_dim = 7
    document_encoder_dim = 9
    emb_matrix = torch.rand(10, 5)

    doc = torch.LongTensor([
        [[4, 3, 3], [1, 2, 3], [5, 2, 1], [7, 1, 3]],
        [[0, 0, 0], [8, 2, 1], [6, 5, 9], [7, 1, 3]]
    ])

    model = HierarchicalEncoder(emb_matrix,
                                False,
                                sentence_encoder_dim,
                                document_encoder_dim,
                                4,
                                3,
                                'attention',
                                nn.GRU,
                                class_weights)

    output = model(doc, '', '')
    # log_msg(output)
    print_msg(output.size())

    """
    doc = " I am a student. 



    """


if __name__ == '__main__':
    debug = True
    test_model()
