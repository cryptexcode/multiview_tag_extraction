import sys
sys.path.append('../')

from get_predictions import parameters as P
import torch

torch.manual_seed(7)
if torch.cuda. \
        is_available():
    torch.cuda.manual_seed_all(7)

import json
import numpy as np
import pandas as pd
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from experiments import exp_utils as U
from torch_helper import TorchHelper
from prepare_data.dataset_loader import MpstDataset
from experiments.evaluation_reports import EvaluationReports
from get_predictions.integrated_hierarchical_model import IntegratedHierarchicalModel
from prepare_data import bert_feature_loader, elmo_feature_loader
import config as C
import warnings

warnings.filterwarnings('ignore')
torch_helper = TorchHelper()
EvaluationReports = EvaluationReports()

target_classes = 71
top_n_list = [1, 3, 5, 8, 10]

batch_size = P.batch_size

collect_attention = True

output_dir_path = '../../results/glove/' + \
                  '13[sgd_lowlr]_run_lr_0.2_Narr_Rev_G_res_s_mlp_glove__l2_0.15/'

# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------
# Load Data using the data generator
root_data_path = '../../data/'

id_title_tag_df = pd.read_csv(root_data_path + '/MPST/id_title_tags.tsv', sep='\t')

# Loads a dictionary like {1:murder, 2: violence ....}
index_to_tag_dict = json.load(open(C.processed_data_dump_path + 'index_to_tag.json', 'r'))

# Load Partition Information
partition_dict = json.load(open(root_data_path + '/MPST/partition.json', 'r'))
train_id_list, val_id_list, test_id_list = partition_dict['train'], partition_dict['val'], \
                                               partition_dict['test']

print('Data Split: Train (%d), Dev (%d), Test (%d)' % (len(train_id_list), len(val_id_list), len(test_id_list)))

# Create the data loaders for all splits
plot_sequence_dict = {}

""" Default setup of embedding """
data_loading_system = MpstDataset
if P.pretrained_embedding_mode == 'bert':
    data_loading_system = bert_feature_loader.BertFeatureLoader
elif P.pretrained_embedding_mode == 'elmo':
    data_loading_system = elmo_feature_loader.DataLoaderForElmo

# training_set = data_loading_system(train_id_list)
# train_generator = data.DataLoader(training_set, batch_size, shuffle=True, collate_fn=training_set.collation_method,
#                                   num_workers=C.dataloader_workers)
# print('Train Loaded')

validation_set = data_loading_system(val_id_list)
val_generator = data.DataLoader(validation_set, batch_size, shuffle=True, collate_fn=collation_method,
                                num_workers=C.dataloader_workers)
print('Validation Loaded')

#test_set = data_loading_system(test_id_list)
#test_generator = data.DataLoader(test_set, batch_size, shuffle=True, collate_fn=test_set.collation_method,
#                                 num_workers=C.dataloader_workers)
#print('Test Loaded')

# rank_hit_loss = RankHitLoss()

# print(len(train_generator), len(val_generator), len(test_generator))


def compute_loss(y_true, y_pred, return_var=False):
    # KL Divergence
    loss = F.kl_div(torch.log(y_pred), y_true)

    if return_var:
        return loss

    return loss.data.item()


# ------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def create_model():
    """
    Creates and returns the EmotionFlowModel.
    Moves to GPU if found any.
    :return:

    """
    embedding_matrix = None
    if P.pretrained_embedding_mode == 'glove':
        embedding_matrix = torch.Tensor(json.load(
            open(C.processed_data_dump_path + 'pretrained_' + P.pretrained_embedding_mode + '.json', 'r'))).float()

    model = IntegratedHierarchicalModel(embedding_matrix,
                                        freeze_embedding=False,
                                        sentence_encoder_dim=32,
                                        document_encoder_dim=32,
                                        pooling_method='attention',
                                        encoder_type=nn.LSTM,
                                        batch_size=batch_size,
                                        share_embedding=True,
                                        feature_type=P.pretrained_embedding_mode)

    return model


def inference(model, data_generator):
    """

    :param model: Pytorch model
    :param data_generator: Data Generator to provide data to the model
    :return: List of imdb ids and List of output probabilities
    """
    predicted_probabilities_list = []
    imdb_ids_list = []
    tag_indices_list = []
    one_hot_vector_list = []

    """ Attention Weights"""
    plot_doc_attention_weights = []
    plot_sent_attention_weights = []
    rb_doc_attention_weights = []
    rb_sent_attention_weights = []
    rt_doc_attention_weights = []
    rt_sent_attention_weights = []
    gate_weights = []
    sent_to_tag_att_weights = []

    model.eval()

    with torch.no_grad():

        for batch_idx, input_data in enumerate(data_generator):
            if P.pretrained_embedding_mode == 'elmo':
                imdb_id, narrative_sequence, review_sequence, narrative_mask, review_mask, tag_list, tag_indices, \
                target_one_hot_vector = input_data
            elif P.pretrained_embedding_mode == 'glove':
                imdb_id, narrative_sequence, review_sequence, tag_list, tag_indices, target_one_hot_vector = input_data

                if torch.cuda.is_available():
                    narrative_sequence = narrative_sequence.cuda()
                    review_sequence = review_sequence.cuda()
                narrative_mask, review_mask = None, None

            # if C.pretrained_embedding_mode != 'elmo':
            #    narrative_sequence.cuda()
            #    review_sequence.cuda()

            model_output_dict = model(narrative_sequence, review_sequence, narrative_mask, review_mask)

            # Accumulate data
            predicted_probabilities_list.append(model_output_dict['out'].cpu())

            # If model returns attention weights, Collect them
            if collect_attention:
                if model_output_dict['plot_doc_attn'] is not None:
                    plot_doc_attention_weights.extend(model_output_dict['plot_doc_attn'].cpu().data.numpy())
                if model_output_dict['plot_sent_attn'] is not None:
                    plot_sent_attention_weights.extend(model_output_dict['plot_sent_attn'].cpu().data.numpy())
                if model_output_dict['review_body_doc_attn'] is not None:
                    rb_doc_attention_weights.extend(model_output_dict['review_body_doc_attn'].cpu().data.numpy())
                if model_output_dict['review_body_sent_attn'] is not None:
                    rb_sent_attention_weights.extend(model_output_dict['review_body_sent_attn'].cpu().data.numpy())
                if model_output_dict['review_title_doc_attn'] is not None:
                    rt_doc_attention_weights.extend(model_output_dict['review_title_doc_attn'].cpu().data.numpy())
                if model_output_dict['review_title_sent_attn'] is not None:
                    rt_sent_attention_weights.extend(model_output_dict['review_title_sent_attn'].cpu().data.numpy())
                if model_output_dict['gate_weights'] is not None:
                    gate_weights.extend(model_output_dict['gate_weights'].cpu().data.numpy().tolist())
                if model_output_dict['sent_to_tag_weighted'] is not None:
                    sent_to_tag_att_weights.extend(model_output_dict['sent_to_tag_weighted'].cpu().data.numpy().tolist())

            one_hot_vector_list.append(target_one_hot_vector)
            imdb_ids_list.extend(imdb_id)
            tag_indices_list.extend(tag_indices)

    # Flatten the output probabilities list
    predicted_probabilities_list = torch.cat(predicted_probabilities_list, dim=0)
    one_hot_vector_list = torch.cat(one_hot_vector_list, dim=0)

    # Place attention weights in a dictionary
    attention_weight_dict = {
        'plot_doc_attention_weights': plot_doc_attention_weights,
        'plot_sent_attention_weights': plot_sent_attention_weights,
        'rb_doc_attention_weights': rb_doc_attention_weights,
        'rb_sent_attention_weights': rb_sent_attention_weights,
        'rt_doc_attention_weights': rt_doc_attention_weights,
        'rt_sent_attention_weights': rt_sent_attention_weights,
        'gate_weights': gate_weights,
        'sent_to_tag_att_weights': sent_to_tag_att_weights
    }

    return imdb_ids_list, tag_indices_list, predicted_probabilities_list, one_hot_vector_list, attention_weight_dict


# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, data_generator):
    """

    :param model:
    :param data_generator:
    :return: average loss {float},
    """

    imdb_ids_list, all_tag_indices, \
    predicted_probabilities_list, \
    target_one_hot_list, \
    attention_weights = inference(model, data_generator)

    sorted_probabilities, sorted_tag_idx = torch.sort(predicted_probabilities_list, 1, descending=True)

    avg_loss = compute_loss(target_one_hot_list, predicted_probabilities_list)

    results = EvaluationReports.get_f1_and_tl(np.array(predicted_probabilities_list).squeeze(),
                                              np.array(target_one_hot_list).squeeze(),
                                              top_n_list)

    mean_rank_hit_score, all_rank_hit_score = EvaluationReports.get_rank_hit_score(all_tag_indices,
                                                                                   predicted_probabilities_list)

    return avg_loss, \
           results, \
           mean_rank_hit_score, \
           all_rank_hit_score, \
           imdb_ids_list, \
           predicted_probabilities_list, \
           sorted_tag_idx, \
           target_one_hot_list, \
           attention_weights


def test(data_generator, model=None, epoch=-1, split=''):
    if model is None:
        model = create_model()
        checkpoint = torch.load(output_dir_path + '/best.pth', map_location = {'cuda:0': 'cpu'})
        model.load_state_dict(checkpoint['model_state'])

    """ Test Data """
    test_loss, \
    test_results, \
    test_mean_rank_hit, \
    test_all_rank_hit, \
    test_imdb_ids_list, \
    test_predicted_probabilities_list, \
    test_sorted_tag_idx, \
    test_target_one_hot_list, \
    test_attention_weights, \
        = evaluate(model, data_generator)

    test_res_str = ', '.join([str(v) for v in test_results])
    with open(output_dir_path + '{}_result(N).txt'.format(split), 'w') as wfr:
        wfr.write('Best Test Results as epoch {}\n'.format(epoch + 1))
        wfr.write('test Rank Hit {}\n'.format(test_mean_rank_hit))
        wfr.write('test Loss {}\n'.format(test_loss))

        print(test_results)
        for i, n in enumerate(top_n_list):
            wfr.write(str(i) + ' ' + str(n) + '----\n')
            wfr.write(str(test_results[i][0]) + ' tl : ' + str(test_results[i][1]))
            wfr.write('\n')

    # Dump prediction probabilities
    json.dump(test_predicted_probabilities_list.numpy().tolist(),
              open(output_dir_path + '{}_predictions_best(N).json'.format(split), 'w'))

    # Write predicted tags
    test_prediction_df_w = U.get_df_predictions_with_ground_truths(test_imdb_ids_list,
                                                                   test_sorted_tag_idx.numpy())
    test_prediction_df_w.to_csv(output_dir_path + '{}_predictions_best(N).csv'.format(split), sep=',', index=False)

    # Write Attention Weights
    test_attn_json = U.format_and_dump_attention_weights(test_imdb_ids_list, test_attention_weights,
                                                         id_title_tag_df, test_prediction_df_w, test_all_rank_hit)
    with open(output_dir_path + '{}_attention_weights(N).json'.format(split), 'w') as f:
        json.dump(test_attn_json, f)

    # # Training set
    # loss, \
    # results, \
    # mean_rank_hit, \
    # all_rank_hit, \
    # imdb_ids_list, \
    # predicted_probabilities_list, \
    # sorted_tag_idx, \
    # target_one_hot_list, \
    # attention_weights  = evaluate(model, val_generator)
    #
    # print(loss, results, mean_rank_hit)
    # U.get_df_predictions_with_ground_truths(imdb_ids_list, sorted_tag_idx.numpy().tolist()).to_csv(
    #     output_dir_path + 'validation_predictions.csv', sep=',', index=False)
    #
    # loss, \
    # results, \
    # mean_rank_hit, \
    # all_rank_hit, \
    # imdb_ids_list, \
    # predicted_probabilities_list, \
    # sorted_tag_idx, \
    # target_one_hot_list, \
    # attention_weights = evaluate(model, train_generator)
    #
    # print(loss, results, mean_rank_hit)
    # U.get_df_predictions_with_ground_truths(imdb_ids_list, sorted_tag_idx.numpy().tolist()).to_csv(
    #     output_dir_path + 'train_predictions.csv', sep=',', index=False)


if __name__ == '__main__':

    test(val_generator, split='val')
    # test(test_generator, split='test')
