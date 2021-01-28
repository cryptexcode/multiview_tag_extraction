import torch
from torch import nn
from torch.nn import functional as F
from models.hierarchical_encoder import HierarchicalEncoder, DocumentEncoderRNN
from models.attention import Attention
import json
import parameters as P
import config as C

debug = False
bidirectional = True


def print_msg(*msg):
    if debug:
        print(msg)


class SBertHierarchicalModel(nn.Module):
    """
    The integrated model that encodes narratives and reviews and uses gating mechanism to combine them.


    """

    def __init__(self,
                 sentence_vector_dim,
                 document_encoder_dim,
                 batch_size):
        """

        Parameters
        ----------
        sentence_vector_dim: Integer
            Dimension of encoded sentences

        document_encoder_dim: Integer
            Determines if the embedding matrix should be tuned or kept frozen

        batch_size: Integer
            Batch size
        """

        super(SBertHierarchicalModel, self).__init__()

        """ Preset values """
        self.linear_layer1_dim = 128
        self.linear_layer2_dim = 64
        self.output_dim = 71

        self.sentence_vector_dim = sentence_vector_dim
        self.document_encoder_dim = document_encoder_dim

        """ Encoder for the narratives """
        if P.NARRATIVE_ACTIVE:
            self.narrative_encoder = nn.LSTM(self.sentence_vector_dim, self.document_encoder_dim, bidirectional=True,
                                             batch_first=True)
            self.narrative_attention = Attention(self.document_encoder_dim * 2)

        """ Encoder for the reviews """
        if P.REVIEW_ACTIVE:
            self.review_encoder = nn.LSTM(self.sentence_vector_dim, self.document_encoder_dim, bidirectional=True,
                                             batch_first=True)
            self.review_attention = Attention(self.document_encoder_dim * 2)

        """ Module for combining the narratives and reviews """
        if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
            if P.GATED:
                self.representation_combiner = RepresentationCombiner2('gate', self.document_encoder_dim * 2)
            else:
                self.representation_combiner = RepresentationCombiner('simple_concat', self.document_encoder_dim * 2)

        """ Fully Connected layer """
        if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
            if P.GATED:
                fc_input_dim = self.document_encoder_dim * 2
            else:
                fc_input_dim = self.document_encoder_dim * 4
        else:
            fc_input_dim = self.document_encoder_dim * 2

        if P.USE_RESIDUAL:
            if P.NARRATIVE_ACTIVE and P.REVIEW_ACTIVE:
                fc_input_dim *= 3
            else:
                fc_input_dim *= 2

        print_msg('fc_input_dim 2', fc_input_dim)
        if P.SENT_LEVEL_PRED:
            fc_input_dim += self.output_dim
            self.sent_to_pred_linear = nn.Linear(self.sentence_vector_dim, self.output_dim)
            if P.SENT_LEVEL_POOL == 'attention':
                self.sent_to_pred_att = Attention(self.output_dim)

        if P.REVIEW_ACTIVE and P.SENT_LEVEL_PRED_REVIEW:
            fc_input_dim += self.output_dim
            self.sent_to_pred_linear_review = nn.Linear(self.sentence_vector_dim, self.output_dim)
            if P.SENT_LEVEL_POOL == 'attention':
                self.sent_to_pred_att_review = Attention(self.output_dim)
        print_msg('fc_input_dim 3', fc_input_dim)

        if P.SMALL_MLP:
            self.fc_out = nn.Sequential(
                nn.Dropout(P.DROPOUT_RATE),  # New addition in 8
                nn.Linear(fc_input_dim, self.output_dim)
            )
            print('small mlp')
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
            print('larger mlp')

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
        batch_size = len(narrative_features)

        """ Run encoding on narrative and review """
        if P.NARRATIVE_ACTIVE:
            out, hidden = self.narrative_encoder(narrative_features, (self.init_hidden(batch_size), self.init_hidden(batch_size)))

            narrative_vector, narrative_doc_attention, _ = self.narrative_attention(out, narrative_mask)
            print_msg('N', narrative_vector.size())

        if P.REVIEW_ACTIVE:
            out, hidden = self.review_encoder(review_features, (self.init_hidden(batch_size), self.init_hidden(batch_size)))

            review_vector, review_doc_attention, _ = self.review_attention(out, review_mask)
            print_msg('R', review_vector.size())

        """ [Conditional] Combine representations """
        if P.REVIEW_ACTIVE and P.NARRATIVE_ACTIVE:
            gate_weights, combined_representation = self.representation_combiner([narrative_vector, review_vector])

            # cl = combined_representation * review_mask
            # cl2 = F.sigmoid(narrative_vector) * (1. - review_mask)
            #
            # combined_representation = cl + cl2
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
            print_msg('sent_to_tag_prob size', narrative_features.size(),
                      narrative_mask.unsqueeze(2).size())
            sent_to_tag_prob = self.sent_to_pred_linear(
                narrative_features) * narrative_mask.unsqueeze(2)

            # Earlier another attention was computed separately. Now using the doc attention
            # 32 15 71 X  32 15 (NEW)
            sent_to_tag_weighted = sent_to_tag_prob * narrative_doc_attention.unsqueeze(2)
            sent_to_tag_pooled = torch.sum(sent_to_tag_weighted, dim=1)

            combined_representation = torch.cat([combined_representation, sent_to_tag_pooled], -1)

        if P.REVIEW_ACTIVE and P.SENT_LEVEL_PRED_REVIEW:
            sent_to_tag_prob_review = self.sent_to_pred_linear_review(
                review_features) * review_mask.unsqueeze(2)
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
                    'plot_doc_attn': narrative_doc_attention,
                    'review_body_doc_attn': review_doc_attention,
                    'gate_weights': gate_weights,
                    'sent_to_tag_weighted': sent_to_tag_weighted
                    }

        return out_dict

    def init_hidden(self, seq_len):
        h = torch.zeros(2, seq_len, self.document_encoder_dim)

        if torch.cuda.is_available():
            h = h.cuda()

        return h


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
            rb_sig_activated = ((self.incoming_data_size - torch.sum(z > 0.5, dim=-1)).float() / self.incoming_data_size) * 100
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
            rb_sig_activated = ((self.incoming_data_size - torch.sum(z > 0.5, dim=-1)).float() / self.incoming_data_size) * 100
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