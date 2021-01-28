import torch
from torch import nn
from models.document_encoder_rnn import DocumentEncoderRNN
import parameters as P

debug = False

def print_msg(*msg):
    if debug:
        print(msg)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for specific type of text (plot, review title, or review body)

    """

    def __init__(self,
                 embedding_matrix,
                 freeze_embedding,
                 sentence_encoder_dim,
                 document_encoder_dim,
                 pooling_method,
                 encoder_type,
                 batch_size,
                 embedder=None,
                 feature_type='glove'):

        super(HierarchicalEncoder, self).__init__()

        self.sentence_encoder_dim = sentence_encoder_dim
        self.document_encoder_dim = document_encoder_dim

        """ Modules """
        if feature_type == 'glove':
            if not embedder:
                self.embedder = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_embedding)
            else:
                self.embedder = embedder

            self.embedding_dropout = nn.Dropout(P.DROPOUT_RATE)
            self.emb_dim = embedding_matrix.size()[-1]

        elif feature_type == 'elmo':
            self.embedder = embedder
            self.emb_dim = P.ELMO_EMBEDDING_DIMENSION

        self.feature_type = feature_type

        """ Sentence Encoder """
        self.sentence_encoder = DocumentEncoderRNN(self.emb_dim,
                                                   sentence_encoder_dim,
                                                   # max_sent_len,
                                                   pooling_method,
                                                   encoder_type,
                                                   batch_size)
        self.sentence_dropout = nn.Dropout(P.DROPOUT_RATE)

        """ Document Encoder """
        self.document_encoder = DocumentEncoderRNN(sentence_encoder_dim * 2,
                                                   document_encoder_dim,
                                                   # max_doc_len,
                                                   pooling_method,
                                                   encoder_type,
                                                   batch_size)
        self.document_dropout = nn.Dropout(P.DROPOUT_RATE)

        """ Training utils"""
        self.sent_batchnorm = nn.BatchNorm1d(self.sentence_encoder_dim * 2)
        self.doc_batchnorm = nn.BatchNorm1d(self.document_encoder_dim * 2)

    def forward(self, doc_matrix, word_level_mask=None):
        """
        Parameters
        ----------

        doc_matrix: torch.LongTensor
            The document matrix of size B * M * N. Where,
                B: Batch size
                M: Longest document length in terms of number of sentences
                N: Longest sentence length in terms of number of words

        # embedder: torch.nn.Embedding
        # doc_mask: torch.nn.LongTensor
        """

        if self.feature_type == 'glove':
            """ Create Masks """
            current_batch_size, max_doc_len, max_sent_len = doc_matrix.size()[:3]
            mask_matrix = (doc_matrix > 0).float()
            word_level_mask = mask_matrix.view(-1, max_sent_len)
            sentence_level_mask = (torch.sum(word_level_mask, dim=-1) > 0).float().view(-1, max_doc_len)
            # print_msg(sentence_level_mask)
            # print('MASK', mask_matrix.size(), word_level_mask.size(), sentence_level_mask.size())

            """ B * M * N * d """
            embedded_matrix = self.embedding_dropout(self.embedder(doc_matrix))

        elif self.feature_type == 'elmo':
            embedded_matrix, word_level_mask, sentence_level_mask = doc_matrix
            embedded_matrix = embedded_matrix.cuda(device=torch.device('cuda:0'))
            word_level_mask = word_level_mask.cuda(device=torch.device('cuda:0'))
            sentence_level_mask = sentence_level_mask.cuda(device=torch.device('cuda:0'))
            current_batch_size, max_doc_len, max_sent_len = embedded_matrix.size()[:3]

        # print(embedded_matrix)
        print_msg('embedded matrix', embedded_matrix.size())

        print_msg(embedded_matrix.view(-1, max_doc_len, self.emb_dim).size())

        """ B * M * N * E_s """
        encoded_sentences, sentence_attentions = self.sentence_encoder(embedded_matrix.view
                                                     (-1, max_sent_len, self.emb_dim), word_level_mask.reshape(-1, max_sent_len))  # B * M * rnn_units

        """ Batch Normalization"""
        encoded_sentences = self.sent_batchnorm(encoded_sentences)
        encoded_sentences = self.sentence_dropout(encoded_sentences)

        if P.BIDIRECTIONAL:
            encoded_sentences = encoded_sentences.view(-1, max_doc_len, self.sentence_encoder_dim * 2)
        else:
            encoded_sentences = encoded_sentences.view(-1, max_doc_len, self.sentence_encoder_dim)

        print_msg('encoded sentences', encoded_sentences.size())

        """ Document Encoder"""
        if torch.cuda.is_available():
            sentence_level_mask = sentence_level_mask.cuda()
        encoded_documents, document_attentions = self.document_encoder(encoded_sentences, sentence_level_mask)

        """ Batch Normalization"""
        #encoded_documents = self.doc_batchnorm(encoded_documents)
        encoded_documents = self.document_dropout(encoded_documents)

        if P.BIDIRECTIONAL:
            encoded_documents = encoded_documents.view(-1, self.document_encoder_dim * 2)

        print_msg('Document shape', encoded_documents.size())

        return encoded_documents, \
            document_attentions.view(-1, max_doc_len),\
            sentence_attentions.view(-1, max_doc_len, max_sent_len), \
            encoded_sentences, \
            sentence_level_mask

