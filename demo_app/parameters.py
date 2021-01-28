""" Experiment Configuration """
GPU_ID              = '1'           # --- Important Param ----
RUN_ID              = '1_extralayergate512relu'
RUN_MODE            = 'run'  # run / test/ resume / test_resume
# Fix the path for resume
# run_mode          = 'test_resume'
EXP_NAME            = 'glove_reproduce_sgd_0.2'  # Folder name
MODEL_TYPE          = 'default'  # Supported: default/bert/roberta/xlnet/h_transformer/distilledbert/lm_glove_mix

""" Data Source """
COMBINE_TEXTS       = False
NARRATIVE_ACTIVE    = True          # --- Important Param ---
REVIEW_ACTIVE       = True          # --- Important Param ----NARRATIVE_FIELD
DATA_AUG            = False
LONG_TEXT_POLICY    = 'head_tail'        # head/tail/head_tail/sliding/divide
LONG_TEXT_LEN       = 510
HEAD_LEN            = 128
TAIL_LEN            = 382
SLIDING_CONTEXT_LEN = 20
TUNE_LM             = True
SPECIAL_TOKENS_1    = {'default': 1, 'bert': 101, 'roberta': 0, 'h_transformer': 0, 'distilledbert': 101}
SPECIAL_TOKENS_2    = {'default': 2, 'bert': 102, 'roberta': 2, 'h_transformer': 2, 'distilledbert': 102}
PAD_VALUE           = {'default': 0, 'h_transformer': 1}
VOCAB_SIZE          = {'h_transformer': 50265}
Y_DIM               = 71


""" Model Parameters """
GATED               = True          # --- Important Param ----
USE_RESIDUAL        = False          # --- Important Param ----
SMALL_MLP           = True          # --- Important Param ----
BIDIRECTIONAL       = True
SENT_LEVEL_PRED     = True          # --- Important Param ----
SENT_LEVEL_PRED_REVIEW = True
SHARED_ENCODER      = False         # --- Important Param ----
FREEZE_EMBEDDING    = False
SENT_LEVEL_POOL     = 'attention'   # --- Important Param ---- [options: max, attention]

""" Training Parameters """
LEARNING_RATE       = 0.2          # --- Important Param ----
CLIP_GRAD           = 0.5
WEIGHT_DECAY_VAL    = 0             # --- Important Param ----
OPTIMIZER_TYPE      = "sgd"
LR_SCHEDULE_ACTIVE  = False          # --- Important Param ----
T_MAX               = 5
L2_LAMBDA           = 0.15
L2_REGULARIZE       = True
DROPOUT_RATE        = 0.5

BATCH_SIZE          = 32
BACKPROP_INTERVAL   = 1
MAX_EPOCH           = 50


""" Misc Configuration """
DUMP_TRAIN_RESULT   = False
DUMP_TEST_RESULT    = False
TOP_N_LIST          = [1, 3, 5, 8, 10]
COLLECT_ATTN        = False

""" Feature Configuration """
PRETRAINED_EMBEDDING_MODE   = 'glove'
ELMO_EMBEDDING_DIMENSION    = 256
BERT_EMBEDDING_DIMENSION    = 3072

PADDING_MODE_LEFT = False  # not important
PAD_DIRECTION       = 'left'
SENT_MIN_LEN = 6  # with start and end token

""" Transformer models related Configuration """
HF_MODEL_MAP = {'bert': 'BertModel', 'roberta': 'RobertaModel', 'xlnet': 'XLNetModel',
                'distilledbert': 'DistilBertModel'}
HF_WEIGHT_MAP = {'bert': 'bert-base-cased', 'roberta': 'roberta-base', 'xlnet': 'xlnet-base-cased',
                 'distilledbert': 'distilbert-base-uncased'}

# Must match with HuggingFace ones
BERT_TOKENIZER              = 'BertTokenizer'       # BertTokenizer, XLNetTokenizer, RobertaTokenizer
ROBERTA_TOKENIZER           = 'RobertaTokenizer'
XLNET_TOKENIZER             = 'XLNetTokenizer'
DISTILLED_BERT_TOKENIZER    = 'DistilBertTokenizer'

""" Transformer Config"""
N_ENCODER_LAYER = 1
D_MODEL         = 100
D_FF            = 200
N_ATT_HEAD      = 4


""" Dynamically Create Run Name For Experiments """
if MODEL_TYPE == 'default':
    RUN_NAME = f'{RUN_ID}' \
               f'_{RUN_MODE}_lr_{LEARNING_RATE}' \
               f'{"_Narr" if NARRATIVE_ACTIVE else ""}' \
               f'{"_Rev" if REVIEW_ACTIVE else ""}' \
               f'{"_G" if GATED  else "_C" if REVIEW_ACTIVE else ""}' \
               f'{"_shared_enc" if SHARED_ENCODER and REVIEW_ACTIVE else ""}' \
               f'{"_res" if USE_RESIDUAL and REVIEW_ACTIVE else ""}' \
               f'{"_s_mlp" if SMALL_MLP else "l_mlp"}' \
               f'{"_SPN"  if SENT_LEVEL_PRED else ""}' \
               f'{"_SPR"  if SENT_LEVEL_PRED_REVIEW and REVIEW_ACTIVE else ""}' \
               f'{"_wtdc" + str(WEIGHT_DECAY_VAL) if WEIGHT_DECAY_VAL > 0 else ""}' \
               f'{"_l2" + str(L2_LAMBDA) if L2_REGULARIZE else ""}'
elif MODEL_TYPE == 'h_transformer':
    RUN_NAME = f'{RUN_ID}' \
               f'_{RUN_MODE}_lr_{LEARNING_RATE}' \
               f'{"_Narr" if NARRATIVE_ACTIVE else ""}' \
               f'{"_Rev" if REVIEW_ACTIVE else ""}' \
               f'_enc_layers_{N_ENCODER_LAYER}' \
               f'_dmodel_{D_MODEL}' \
               f'_ffdim_{D_FF}' \
               f'_head_{N_ATT_HEAD}' \
               f'_drop_{DROPOUT_RATE}' \
               f'{"_wtdc" + str(WEIGHT_DECAY_VAL) if WEIGHT_DECAY_VAL > 0 else ""}' \
               f'{"_l2" + str(L2_LAMBDA) if L2_REGULARIZE else ""}'
else:
    RUN_NAME         = f'{RUN_ID}' \
                    f'{"_" + MODEL_TYPE}'\
                    f'{"_Tune" if TUNE_LM else "_NoTune"}'\
                    f'{"_" + LONG_TEXT_POLICY  if  MODEL_TYPE!="default" else ""}' \
                    f'_{RUN_MODE}_lr_{LEARNING_RATE}'\
                    f'{"_Narr"  if  NARRATIVE_ACTIVE    else ""}' \
                    f'{"_G" if GATED else "_C" if REVIEW_ACTIVE else ""}'
        #                     f'{"_Rev"   if  REVIEW_ACTIVE       else ""}' \
#                     f'{"_G"     if  GATED               else "_C"}' \
#                     f'{"_shared_enc"  if  SHARED_ENCODER               else ""}' \
#                     f'{"_res"   if  USE_RESIDUAL        else ""}' \
#                     f'{"_s_mlp"  if SMALL_MLP           else "l_mlp" if  MODEL_TYPE=="default" else ""}'\
#                     f'{"_sent_pred" + SENT_LEVEL_POOL if SENT_LEVEL_PRED else ""}' \
#                     f'{"_wtdc" + str(WEIGHT_DECAY_VAL)  if WEIGHT_DECAY_VAL > 0 else ""}' \
#                     f'{"_l2"+str(L2_LAMBDA) if L2_REGULARIZE else ""}'
#
# # # --> BERT and Other
#
# #
if RUN_MODE == 'resume':
    RUN_NAME = '1_padleft_gate_run_lr_0.2_Narr_Rev_G_s_mlp_SPN_SPR_l20.15'
# # 8: Dropout added before final fc
# # 9: LSTM dim 20
# # 10: LSTM dim 64
#