PROCESSED_DATA_PATH = '../../res/'

idx2w_file_path = PROCESSED_DATA_PATH + 'index_to_word.json'
w2idx_file_path = PROCESSED_DATA_PATH + 'word_to_index.json'


# ----------------------------------------------
# Preprocessing and feature extraction setting
# ----------------------------------------------
should_lowercase = True
sent_start_token, sent_end_token = '<S>', '</S>'
doc_start_token, doc_end_token = 'SOD', 'EOD'
unknown_token, unknown_token_idx = '<UNK>', 3
should_normalize_token = True
digit_placeholder = 'cc'
append_start_end_tokens = True
remove_stops = False


# RAW_DATA_PATH       = RES_DIR + 'data/' + DATASET + '/'
# PROCESSED_DATA_PATH = RES_DIR
# RESULTS_DIR_PATH    = RES_DIR + 'results/' + DATASET + '/'
# FEATURE_DIR_PATH    = RES_DIR + 'feature_files/' + DATASET + '/'
# PARTITION_DICT_PATH = RAW_DATA_PATH + 'partition.json'
# F_PARTITION_DICT_PATH = RAW_DATA_PATH + 'f_partition.json'
#
# if DATASET == 'MPST':
#     root_path = '../../'
#     # processed_data_dump_path = root_path + '/processed_data/mpst/'
#     NARRATIVE_RAW_TEXT_PATH = RAW_DATA_PATH + 'final_plots_wiki_imdb_combined/raw/'
#     REVIEW_SUMMARY_RAW_TEXT_PATH = RAW_DATA_PATH + 'review_summaries_5k/'
#     INSTANCE_TO_TAG_FILE = RAW_DATA_PATH + 'tag_assignment_data/movie_to_label_name.json'
#
#     # Processed data dump directory
#     review_csv_directory_path = root_path + 'data/csv_review_summaries_title_body_combined/'
#     training_imdb_id_path = root_path + 'data/MPST/train_ids.txt'
#     test_id_list_path = root_path + 'data/MPST/test_ids.txt'
#     all_id_list = root_path + 'data/MPST/final_plots_wiki_imdb_combined/imdb_id_list.txt'
#     review_summary_csv_directory_path = '../../data/csv_review_summaries_title_body_combined/'
#
#     TOKENIZED_NARRATIVES_PATH = PROCESSED_DATA_PATH + '/tokenized_dump/imdb_plot/'
#     TOKENIZED_REVIEWS_PATH = PROCESSED_DATA_PATH + '/tokenized_dump/review_summaries/'
#     BPE_NARRATIVES_PATH = PROCESSED_DATA_PATH + '/tokenized_dump/plot_sentences_roberta/'
#     BPE_REVIEWS_PATH = PROCESSED_DATA_PATH + '/tokenized_dump/review_sentences_roberta/'
#     feature_dump_path = FEATURE_DIR_PATH + 'hier_bert_roberta/'
#     sbert_feature_dump_path = FEATURE_DIR_PATH + 'sbert_features/'
#     idx_to_tag_path = PROCESSED_DATA_PATH + '/index_to_tag.json'
#
#     # Pretrained vectors path
#     word2vec_path = '/home/sk/SK/Works/resources/GoogleNews-vectors-negative300.bin.gz'
#     fasttext_path = '/home/sk/SK/Works/resources/fasttext/wiki.en/wiki.en.bin'
#     glove_path = '../../glove.840B.300d.txt'
#
#     # Embedding
#     elmo_options = '../../elmo_options.json'
#     elmo_weights = '../../elmo_weights.hdf5'
#
#
#     vocabulary_path = PROCESSED_DATA_PATH + 'filtered_word_frequency_counts_doc.txt'
#
#     dataloader_workers = 0
#     # hierarchical settings
#     NARRATIVE_MAX_SENTS, NARRATIVE_MAX_TOKENS = 150, 50 # 200, 60
#     REVIEW_MAX_SENTS, REVIEW_MAX_TOKENS = 40, 25
#     # max_rb_doc_len, max_rb_sent_len = 120, 50
#     review_doc_max_sent, review_sent_max_token = 160, 60
#

#
#     # BERT Configuration
#     bert_model = 'bert-base-uncased'
#     bert_do_lowercase = True
#     sentence_pad_element = ''
#     doc_pad_element = []
#     padding_direction = ' right'
#     bert_layer_indexes = [-1,-2,-3,-4]
#     sbert_feature_dim = 768
