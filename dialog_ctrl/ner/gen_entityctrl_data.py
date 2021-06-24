
from src.config import get_params
from transformers import AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os

wn_lemma = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words.append("n't")
stop_words.append("'s")
punctuations = list(string.punctuation)
punctuations.append("``")
punctuations.append("''")

stop_words_and_punctuations = stop_words + punctuations
stop_words_and_punctuations_table = {word: True for word in stop_words_and_punctuations}

label_set = ["O", "B", "I"]

def read_data(input_datapath):
    data = []
    print("Reading data from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            length = len(splits)
            assert length == 2 or length == 4

            # length is 2: dialog context + response
            # length is 4: dialog context + topic + control sentence + response
            if length == 2:
                # dialog context + response
                data.append(line)
            else:
                # only need dialog context + control sentence + response
                data.append(splits[0] + "\t" + splits[2] + "\t" + splits[3])

    return data


def write_data(output_datapath, output_data):
    print("Writing data to %s" % output_datapath)
    with open(output_datapath, "w") as fw:
        for data_sample in output_data:
            fw.write(data_sample + "\n")


def detect_entities(tokenizer, ner_model, sentence):
    tokens = sentence.split()
    token_ids, first_tok_masks = [tokenizer.cls_token_id], [0]
    for token in tokens:
        subs_ = tokenizer.tokenize(token)
        assert len(subs_) > 0
        
        token_ids.extend(tokenizer.convert_tokens_to_ids(subs_))
        first_tok_masks.extend([1] + [0] * (len(subs_) - 1))
    
    token_ids.append(tokenizer.sep_token_id)
    first_tok_masks.append(0)
    
    token_ids = torch.LongTensor([token_ids]).cuda()
    predictions = ner_model(token_ids)

    predictions = predictions[0].data.cpu().numpy() # (seq_len, 3)
    pred_ids = list(np.argmax(predictions, axis=1))

    assert len(pred_ids) == len(first_tok_masks)
    preds_for_each_word = []
    for pred_id, mask in zip(pred_ids, first_tok_masks):
        if mask == 1:
            preds_for_each_word.append(label_set[pred_id])

    assert len(preds_for_each_word) == len(tokens)

    # extract entities
    entity_list = []
    temp = []
    for i, (token, pred) in enumerate(zip(tokens, preds_for_each_word)):
        if pred == "O":
            if len(temp) > 0:
                entity_list.append(" ".join(temp))
                temp = []
        else: 
            # pred == "B" or pred == "I"
            temp.append(token)

    return entity_list


def generate_entity_control_data(tokenizer, ner_model, input_data):
    # aim to generate:
    # dialog context + entity control code (optional) + relevant control sentence (contain entity) + response
    
    output_data = []
    ## TODO
    n_skip, n_skip_no_overlap, n_skip_one_contain_another = 0, 0, 0
    n_control, n_entity_control, n_overlap_control = 0, 0, 0
    total_num_control_code = 0
    for sample_idx, data_item in enumerate(tqdm(input_data)):
        # # Debug only
        # if sample_idx > 1000:
        #     break

        # 1. detect entities for dialog context, control sentence and response
        splits = data_item.split("\t")
        if len(splits) == 2:
            output_data.append(data_item)
            continue
        assert len(splits) == 3
        
        last_turn = splits[0].split(" [SEP] ")[-1]
        control_sent = splits[1]
        response = splits[2]

        if control_sent in response or response in control_sent:
            # if the whole control_sent is a part of response or vise versa, skip this data sample 
            n_skip += 1
            n_skip_one_contain_another += 1
            continue

        last_turn_entities = detect_entities(tokenizer, ner_model, last_turn)
        control_sent_entities = detect_entities(tokenizer, ner_model, control_sent)
        response_entities = detect_entities(tokenizer, ner_model, response)

        # 2. generate control code:
        # 2.1 If there is one or more than one common entity in last_turn, control sentence and response. No need to use entity as control.
        # 2.2 If the entity only exists in control sentence and response, use this as the control code.
        # 2.3 If there is no overlaped entity or words between control sentence and response, skip this data sample.
        # 2.4 If there is no overlapped entity but there are overlapped words, add entity in the control sentence (if any) as the control code if it is not in the dialog context

        # TODO
        # In general, need to trim the control sentence when it is too long.
        # Need to lowercase to match?

        # calculate common entity between control sentence and response
        common_entity_list = []
        for ctrl_entity in control_sent_entities:
            for resp_entity in response_entities:
                if resp_entity in ctrl_entity:
                    common_entity_list.append(ctrl_entity)
                    break
                elif ctrl_entity in resp_entity:
                    common_entity_list.append(resp_entity)
                    break
        
        if len(common_entity_list) == 0:
            # calculate overlap between control sentence and response
            control_word_list = control_sent.split()
            response_word_list = response.split()
            response_word_table = {wn_lemma.lemmatize(word): True for word in response_word_list}
            overlap_phrases = []
            temp = []
            for word in control_word_list:
                if word.lower() in stop_words_and_punctuations_table:
                    continue
                
                if wn_lemma.lemmatize(word) in response_word_table:
                    temp.append(word)
                else:
                    if len(temp) > 0:
                        if len(temp) > 4:
                            temp = temp[:4]
                        overlap_phrases.append(" ".join(temp))
                        temp = []

            if len(overlap_phrases) == 0:
                # skip this data sample
                n_skip += 1
                n_skip_no_overlap += 1
                continue
            
            n_control += 1
            control_code_list = []

            if len(control_sent_entities) > 0:
                n_entity_control += 1
                # reorder control_sent_entities based on the length of the entities (in a reverse order)
                control_sent_entities = sorted(control_sent_entities, key=len, reverse=True)
                for entity in control_sent_entities:
                    if entity not in last_turn:
                        add_flag = True
                        for code in control_code_list:
                            if entity in code:
                                add_flag = False
                                break
                        if add_flag:
                            control_code_list.append(entity)
            else:
                n_overlap_control += 1
                # reorder overlap_phrases based on the length of the phrases (in a reverse order)
                overlap_phrases = sorted(overlap_phrases, key=len, reverse=True)[:3]
                for phrase in overlap_phrases:
                    if phrase not in last_turn:
                        add_flag = True
                        for code in control_code_list:
                            if phrase in code:
                                # remove repeat word
                                add_flag = False
                                break
                        if add_flag:
                            control_code_list.append(phrase)

        else:
            n_entity_control += 1
            n_control += 1
            control_code_list = []
            # reorder common_entity_list based on the length of the entities (in a reverse order)
            common_entity_list = sorted(common_entity_list, key=len, reverse=True)
            for entity in common_entity_list:
                if entity not in last_turn:
                    add_flag = True
                    for code in control_code_list:
                        if entity in code:
                            add_flag = False
                            break
                    if add_flag:
                        control_code_list.append(entity)

        total_num_control_code += len(control_code_list)

        if len(control_code_list) > 0:
            output_data.append(splits[0] + "\t" + " [CTRL] ".join(control_code_list) + "\t" + control_sent + "\t" + response)
        else:
            output_data.append(splits[0] + "\t" + control_sent + "\t" + response)

    avg_num_control_code = total_num_control_code * 1.0 / n_control

    print("number of skip sentences: %d (one contain another: %d + no overlap: %d)" % (n_skip, n_skip_one_contain_another, n_skip_no_overlap))
    print("Total data size: %d. Number of control case: %d (entity control: %d + overlap control: %d)" % (len(output_data), n_control, n_entity_control, n_overlap_control))
    print("Number of control code: %d vs. number of control case: %d (averaged control code per case: %.4f)" % (total_num_control_code, n_control, avg_num_control_code))

    return output_data


def main(params):
    # load model and tokenizer
    model_saved_path = os.path.join(params.saved_folder, params.model_name+".pt")
    ner_model = torch.load(model_saved_path)["model"]
    ner_model.cuda()
    ner_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    # load data
    datafolder = os.path.join(params.default_folder, params.infer_datafolder)
    input_datapath = os.path.join(datafolder, params.infer_dataname)
    output_datapath = os.path.join(datafolder, params.output_dataname)

    # read input data
    input_data = read_data(input_datapath)

    # process data (generate entity control data)
    output_data = generate_entity_control_data(tokenizer, ner_model, input_data)

    # write output data
    write_data(output_datapath, output_data)


if __name__ == "__main__":
    params = get_params()
    main(params)