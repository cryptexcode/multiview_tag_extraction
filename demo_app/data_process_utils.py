import config as C


def convert_text_to_sequence(text_tokenized, word2idx_dict):
    """
    Convert a text into sequence of integers using the given word2idx_dict

    :param text_tokenized: is_valid
    :param word2idx_dict: Dictionary. key: word, value: Integer Index

    :return:
    """
    def filter_tokens(token_list):
        return [w if w in word2idx_dict else C.unknown_token for w in token_list]

    if isinstance(text_tokenized[0], list):
        sentence_list = [filter_tokens(txt) for txt in text_tokenized]
        sequence = [[word2idx_dict[w] for w in filtered_text] for filtered_text in sentence_list]
    else:
        filtered_text = filter_tokens(text_tokenized)
        sequence = [word2idx_dict[w] for w in filtered_text]

    return sequence


def convert_text_to_hierarchical_sequence(tokenized_data, word2idx_dict, sentence_filter_len=0):
    """

    :param tokenized_data:
    :param word2idx_dict:
    :param sentence_filter_len: To avoid short sentences (maybe blank)
    :return:
    """
    sequence = [convert_text_to_sequence(e, word2idx_dict) for e in tokenized_data
                if sentence_filter_len > 0 and len(e) > sentence_filter_len]
    return sequence


def pad_truncate_one_sequence(seq, max_len, direction='right', value=0):
    """


    :param seq:
    :param max_len:
    :param direction:
    :param value:
    :return:
    """
    seq_len = len(seq)
    pad_len = abs(max_len - seq_len)
    padded_sequence = [0] * max_len

    if seq_len < max_len:
        padding = [value] * pad_len
        if direction == 'left':
            padded_sequence = padding + seq
        elif direction == 'right':
            padded_sequence = seq + padding
        else:
            raise ValueError("Check the value of argument 'direction'")

    elif seq_len > max_len:
        # Truncate
        if direction == 'left':
            padded_sequence = [seq[0]] + seq[-(max_len-1):] # -1 is to handle SOD and EOD
        elif direction == 'right':
            padded_sequence = seq[: max_len-1] + [seq[-1]]
        else:
            raise ValueError("Check the value of argument 'direction'")

    else:
        padded_sequence = seq

    return [1] + padded_sequence + [2]


def pad_hierarchical_sequence(seq_list, max_doc_len, max_sent_len, pad_direction='right', sp_tkn1=1, sp_tkn2=2, value=0):
    """
    Parameters:
    -----------
    seq_list : List
    max_sent_len : Integer
    max_doc_len : Integer
    value : Integer (default = 0)

    """

    # sequence of sentences
    padded_sentence_list = [pad_truncate_one_sequence(seq, max_sent_len, pad_direction) for seq in seq_list]

    return padded_sentence_list
