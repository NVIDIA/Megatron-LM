# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Preprocessing for Wizard of Wikipedia and Wizard of Internet datasets"""

import torch
import argparse
from nltk import word_tokenize
from tqdm import tqdm
import numpy as np
import json

def get_args():
    parser = argparse.ArgumentParser(description="Preprocessing")

    parser.add_argument("--func", type=str, default=None,
                        help="choose to run which function")
    parser.add_argument("--raw_file", type=str, default=None,
                        help="path of the input file")
    parser.add_argument("--processed_file", type=str, default=None,
                        help="path of the output file")
    parser.add_argument("--knwl_ref_file", type=str, default=None,
                        help="path of the knowledge reference file")
    parser.add_argument("--resp_ref_file", type=str, default=None,
                        help="path of the knowledge reference file")
    parser.add_argument("--knwl_gen_file", type=str, default=None,
                        help="path of the generated knowledge file")
    parser.add_argument("--test_file", type=str, default=None,
                        help="path of the test file")
    parser.add_argument("--train_file", type=str, default=None,
                        help="path of the train file")
    parser.add_argument("--model_file", type=str, default=None,
                        help="path of the model file")
    parser.add_argument("--data_type", type=str, default=None,
                        help="data types, choose one out of three types: \
                              wow_seen, wow_unseen, and woi")
    parser.add_argument("--seed", type=int, default=1234,
                        help="random seed")

    args = parser.parse_args()
    return args


def process_wow_dataset(raw_file, processed_file, knwl_ref_file, resp_ref_file):
    """
      This is a function used for processing the wizard of wikipedia (wow) dataset
      Expected processed format:
      topic \t dialogue context \t golden knowledge \t golden response
    """

    # loading the raw data
    print("> Loading data from %s" % raw_file)
    with open(raw_file, "r") as fr:
        dialog_data = json.load(fr)
    
    print("> Processing data ...")
    fproc = open(processed_file, "w")
    fknwl = open(knwl_ref_file, "w") if knwl_ref_file else None
    fresp = open(resp_ref_file, "w") if resp_ref_file else None
    
    for i, sample in enumerate(tqdm(dialog_data)):
        # get all the dialog data for a single dialog sample
        dialog = sample["dialog"]
        
        turn_list = []  # collect the dialog history
        # processing for each single dialog sample
        for j, turn in enumerate(dialog):
            # text of each turn
            text = turn["text"]
            if not (text.endswith("?") or text.endswith(".") or text.endswith("!")):
                text = text + "."
            
            if j == 0:
                # first turn
                turn_list.append(text)
                continue

            speaker = turn["speaker"].lower()
            if "wizard" in speaker:
                checked_sentence = list(turn["checked_sentence"].values())  # knowledge
                checked_passage = list(turn["checked_passage"].values())    # topic
                
                assert len(checked_sentence) <= 1

                # get the ground truth knowledge
                if len(checked_sentence) > 0:
                    checked_sentence = checked_sentence[0]
                else:
                    checked_sentence = "no_passages_used"

                if len(checked_passage) == 1:
                    checked_passage = checked_passage[0]
                else:
                    checked_passage = "no_passages_used"

                # get the topic
                if checked_passage != "no_passages_used":
                    topic = checked_passage
                else:
                    topic = sample["chosen_topic"]
                
                dialog_context = " [SEP] ".join(turn_list)
                knowledge = checked_sentence
                response = text
                # add the response into the dialog history
                turn_list.append(response)

                # write to the output files
                fproc.write(topic + "\t" + dialog_context + "\t" + \
                                knowledge + "\t" + response + "\n")
                
                if fknwl:
                    fknwl.write(knowledge + "\n")
                if fresp:
                    # tokenize for evaluation
                    response = " ".join(word_tokenize(response))
                    fresp.write(response + "\n")

            else:
                assert "apprentice" in speaker
                turn_list.append(text)

    fproc.close()
    if fknwl:
        fknwl.close()
    if fresp:
        fresp.close()


def process_woi_dataset(raw_file, processed_file, knwl_ref_file, resp_ref_file):
    """
      This is a function used for processing the wizard of internet (woi) dataset
      Expected processed format:
      topic \t dialogue context \t golden knowledge \t golden response
    """
    
    print("> Processing %s" % raw_file)
    fproc = open(processed_file, "w")
    fknwl = open(knwl_ref_file, "w") if knwl_ref_file else None
    fresp = open(resp_ref_file, "w") if resp_ref_file else None
    
    with open(raw_file, "r") as fr:
        for i, line in tqdm(enumerate(fr)):
            # read line by line, each line uses json format
            line = line.strip()
            item_dict = json.loads(line)

            # item_dict is a dictionary
            # its key is the data id, and its value contains all the data content
            item_dict = item_dict.values()
            item_dict = list(item_dict)[0]  # len(item_dict) == 1
            
            # get the whole dialog data for a single dialog sample
            dialog_data = item_dict['dialog_history']
            length = len(dialog_data)
            
            turn_list = []  # collect the dialog history
            search_text = ""
            for i in range(length):
                item = dialog_data[i]
                action = item['action']

                if action == "Wizard => SearchAgent":
                    search_text = item['text']

                elif action == "Wizard => Apprentice":
                    if len(turn_list) == 0:
                        # first turn
                        turn = item['text']
                        turn_list.append(turn)
                        continue

                    # get the relevant content
                    contents = item["context"]["contents"]
                    selects = item["context"]["selected_contents"]
                    flag = selects[0][0]
                    selects = selects[1:]
                    assert len(selects) == len(contents)
                    
                    # get the topic
                    if flag:
                        # no knowledge sentence is used for the response
                        topic = "no_topic"
                        knwl_sent = "no_passages_used"
                    else:
                        # we consider the search text as the topic
                        topic = search_text
                        # get the knowledge sentence
                        knwl_sent = ""
                        for content, select in zip(contents, selects):
                            content = content['content']
                            assert len(content) == len(select)
                            for c, s in zip(content, select):
                                if s:
                                    knwl_sent = c
                                    break

                    if knwl_sent == "":
                        # no knowledge is used for the response
                        topic = "no_topic"
                        knwl_sent = "no_passages_used"

                    # get dialogue context, knowledge, and response 
                    dialog_context = " [SEP] ".join(turn_list)
                    response = item['text']

                    # processing
                    topic = topic.replace("\n", "").replace("\r", \
                                "").replace("\t", "")
                    dialog_context = dialog_context.replace("\n", "").replace("\r", \
                                "").replace("\t", "")
                    knwl_sent = knwl_sent.replace("\n", "").replace("\r", \
                                "").replace("\t", "")
                    response = response.replace("\n", "").replace("\r", \
                                "").replace("\t", "")
                    
                    if topic != "no_topic":
                        # write to the ouput files
                        fproc.write(topic + "\t" + dialog_context + "\t" + \
                                        knwl_sent + "\t" + response + "\n")
                        if fknwl:
                            fknwl.write(knwl_sent + "\n")
                        if fresp:
                            # tokenize for evaluation
                            response = " ".join(word_tokenize(response))
                            fresp.write(response + "\n")

                    turn_list.append(response)

                elif action == "Apprentice => Wizard":
                    turn = item['text']
                    turn_list.append(turn)

                else:
                    assert action == "SearchAgent => Wizard", \
                            "Please check whether you have used the correct data!"

    fproc.close()
    if fknwl:
        fknwl.close()
    if fresp:
        fresp.close()


def get_database(test_datapath, train_datapath, data_type):
    """Get the database by topics"""

    assert data_type in ["wow_seen", "wow_unseen", "woi"], \
                "Please input a correct data type!!"

    # get test data topic dictionary
    print("> reading test data from %s" % test_datapath)
    test_topics = {}
    with open(test_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            topic = splits[0]
            test_topics[topic] = True

    print("> reading data from %s" % train_datapath)
    train_data_by_topic = {}
    dialog_data_by_topic = {}
    dialog_examples = []
    with open(train_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            topic = splits[0]
            turns = splits[1].split(" [SEP] ")[-3:]
            knowledge = splits[2]
            response = splits[3]
            # filtering data samples
            if knowledge == "no_passages_used":
                # when no knowledge is used
                continue
            if data_type != "wow_seen" and ("(" in knowledge or ")" in knowledge):
                # when bracket exists in the knowledge
                continue
            if data_type != "wow_seen" and topic not in knowledge:
                # when topic does not exist in the knowledge
                continue

            # get the instance
            last_turn = turns[-1]
            instance = "( " + last_turn + " ) " + topic + " => " + knowledge
            
            # construct dialog example
            dialog_example = ""
            if data_type != "wow_seen":
                dialog_example += "( " + topic + " ) "
            for i, turn in enumerate(turns):
                if i != 0:
                    dialog_example += " "
                dialog_example += turn
            
            # check overlaps
            if topic in test_topics:
                if topic not in train_data_by_topic:
                    train_data_by_topic[topic] = [instance]
                else:
                    train_data_by_topic[topic].append(instance)
                
                if topic not in dialog_data_by_topic:
                    dialog_data_by_topic[topic] = [dialog_example]
                else:
                    dialog_data_by_topic[topic].append(dialog_example)
            
            else:
                # filtering data samples
                if len(knowledge.split()) > 20:
                    # knowledge is too long
                    continue
                if knowledge.startswith("It") or knowledge.startswith("it") or \
                   knowledge.startswith("This") or knowledge.startswith("this"):
                    continue
                
            # append all the data into dialogue examples list
            dialog_examples.append((topic, dialog_example, instance))

    return train_data_by_topic, dialog_data_by_topic, dialog_examples


emb_dict = {}
def select_prompts_based_on_similarity(
        query, dialog_list, prompt_list, topic, tokenizer, encoder, topk):
    """Select samples based on the similarity"""

    with torch.no_grad():
        # get the query embeddings
        query_ids = tokenizer.encode(query)
        query_ids = torch.LongTensor([query_ids]).cuda()
        query_emb = encoder(input_ids=query_ids).pooler_output
        query_emb = query_emb[0]
        
        # calculate embeddings for the samples in the database
        if topic in emb_dict:
            example_embeddings = emb_dict[topic]
            example_embeddings = example_embeddings.cuda()
        else:
            for idx, example in enumerate(dialog_list):
                example_ids = tokenizer.encode(example)
                example_ids = torch.LongTensor([example_ids]).cuda()
                example_emb = encoder(input_ids=example_ids).pooler_output
                if idx == 0:
                    example_embeddings = example_emb
                else:
                    example_embeddings = torch.cat(
                        (example_embeddings, example_emb), dim=0)
            emb_dict[topic] = example_embeddings.cpu()

        # compare the similarity and select the topk samples
        similarity_list = example_embeddings.matmul(query_emb)
        _, indices = torch.topk(similarity_list, k=topk)
    
    indices = indices.tolist()
    indices = indices[::-1] # reverse the order
    selected_prompts = []
    for index in indices:
        # index = index.item()
        selected_prompts.append(prompt_list[index])

    return selected_prompts


def prompt_selection_for_knowledge_generation(
        test_datapath, train_datapath, model_path, output_prompt_path, data_type):
    """Selecting prompts for the knowledge generation"""

    print("> Selecting prompts for the knowledge generation")

    train_data_by_topic, dialog_data_by_topic, dialog_examples = \
                            get_database(test_datapath, train_datapath, data_type)
    
    from transformers import DPRQuestionEncoderTokenizer
    print("> loading tokenizer and encoder")
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                    'facebook/dpr-question_encoder-single-nq-base')
    encoder = torch.load(model_path).cuda()

    print("> getting dialog embeddings")
    with torch.no_grad():
        for idx, example in tqdm(enumerate(dialog_examples)):
            dialog = example[1]
            dialog_ids = tokenizer.encode(dialog)
            dialog_ids = torch.LongTensor([dialog_ids]).cuda()
            dialog_emb = encoder(input_ids=dialog_ids).pooler_output

            if idx == 0:
                dialog_embeddings = dialog_emb
            else:
                dialog_embeddings = torch.cat((dialog_embeddings, dialog_emb), dim=0)

    print("> reading test data from %s" % test_datapath)
    prompt_list_for_each_sample = []
    with open(test_datapath, "r") as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()

            splits = line.split("\t")
            topic = splits[0]
            turns = splits[1].split(" [SEP] ")[-3:]

            # get the query sentence
            query_sent = ""
            if data_type != "seen":
                query_sent += "( " + topic + " ) "
            for i, turn in enumerate(turns):
                if i != 0:
                    query_sent += " "
                query_sent += turn

            if topic not in train_data_by_topic:
                # get the query embedding
                query_ids = tokenizer.encode(query_sent)
                query_ids = torch.LongTensor([query_ids]).cuda()
                query_emb = encoder(input_ids=query_ids).pooler_output
                query_emb = query_emb[0]

                # calculate the similarity
                similarity_list = dialog_embeddings.matmul(query_emb)
                _, indices = torch.sort(similarity_list)
                indices = indices.tolist()
                selected_topics = {}
                selected_prompts = []
                num_prompt = 0
                for index in indices:
                    example = dialog_examples[index]
                    topic_temp = example[0]
                    if topic_temp not in selected_topics:
                        selected_topics[topic_temp] = True
                        selected_prompts.append(example[2])
                        num_prompt += 1
                        if num_prompt == 10:
                            break
                
                # get the selected samples
                example_list = selected_prompts[::-1]
                key = topic + " " + turns[-1]
                prompt_list_for_each_sample.append({key: example_list})

            else:
                num_data_sample = min(len(train_data_by_topic[topic]), 10)
                total_example_list = train_data_by_topic[topic]
                
                dialog_list = dialog_data_by_topic[topic]
                assert len(dialog_list) == len(train_data_by_topic[topic])

                # calculate the similarity
                example_list = select_prompts_based_on_similarity(
                                query_sent, dialog_list, total_example_list, 
                                topic, tokenizer, encoder, topk=num_data_sample)
                
                key = topic + " " + turns[-1]
                prompt_list_for_each_sample.append({key: example_list})

    print("writing to %s" % output_prompt_path)
    with open(output_prompt_path, "w") as f:
        for instance in tqdm(prompt_list_for_each_sample):
            json.dump(instance, f)
            f.write("\n")


def prompt_selection_for_response_generation(input_path, output_path, seed):
    """Selecting prompts for the response generation"""

    print("> Selecting prompts for the response generation")
    print("> set random seed")
    np.random.seed(seed)

    prompt_example_list = []
    print("> reading data from %s" % input_path)
    with open(input_path, "r") as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()
            splits = line.split("\t")

            # get the topic, context, knowledge and response
            topic = splits[0]
            dialog_context = splits[1]
            knowledge = splits[2]
            response = splits[3]
            turns = dialog_context.split(" [SEP] ")[-3:]
            if knowledge == "no_passages_used":
                continue

            # calculate the overlap ratio
            from nltk import word_tokenize
            knowledge_sent_token_list = word_tokenize(knowledge)
            knowledge_sent_token_dict = {token: True for token in knowledge_sent_token_list}
            knowledge_len = len(knowledge_sent_token_list)
            response_token_list = word_tokenize(response)
            response_len = len(response_token_list)
            num_overlap_token = 0
            accumulator = 0
            for token in response_token_list:
                if token in knowledge_sent_token_dict:
                    accumulator += 1
                else:
                    if accumulator >= 10:
                        num_overlap_token += accumulator
                    accumulator = 0
            if accumulator >= 10:
                num_overlap_token += accumulator
            
            # filtering the data based on the ratio
            if num_overlap_token > response_len * 0.9 or num_overlap_token < response_len * 0.6:
                continue
            if num_overlap_token < knowledge_len * 0.8:
                continue
            
            last_turn = " ".join(word_tokenize(turns[-1]))
            knowledge = " ".join(word_tokenize(knowledge))
            response = " ".join(word_tokenize(response))
            prompt_example = ""
            # add dialog context
            prompt_example += "Topic: " + topic + ". "
            prompt_example += "User says: " + last_turn + " "
            prompt_example += "We know that: " + knowledge + " "
            prompt_example += "System replies: " + response
            
            prompt_example_list.append(prompt_example)
        
    # shuffle the prompt examples
    np.random.shuffle(prompt_example_list)
    
    print("> writing to %s" % output_path)
    with open(output_path, "w") as f:
        # f.write("Generate the System's response based on the knowledge sentence:\n")
        for i in tqdm(range(20)):
            example = prompt_example_list[i]
            f.write(example + "\n")


def prepare_input_for_response_generation(test_file, knwl_gen_file, processed_file):
    """Preparing inputs for the response generation"""

    print("> Reading knowledge file from %s" % knwl_gen_file)
    # get the knowledge list
    with open(knwl_gen_file, "r") as f:
        knowledge_list = f.readlines()
    
    print("> Processing ...")
    with open(test_file, "r") as fr:
        with open(processed_file, "w") as fw:
            for line_num, line in enumerate(tqdm(fr)):
                line = line.strip()
                splits = line.split("\t")
                # prepare topic, context, knowledge and response
                topic = splits[0]
                dialog_context = splits[1]
                response = splits[3]
                knowledge = knowledge_list[line_num]
                knowledge = knowledge.strip()
                if "<|endoftext|>" in knowledge:
                    knowledge = knowledge.replace("<|endoftext|>", "")

                # write to the output file
                fw.write(topic + "\t" + dialog_context + "\t" \
                                     + knowledge + "\t" + response + "\n")


if __name__ == "__main__":

    args = get_args()
    if args.func == "process_wow_dataset":
        process_wow_dataset(args.raw_file, args.processed_file, args.knwl_ref_file, args.resp_ref_file)

    elif args.func == "process_woi_dataset":
        process_woi_dataset(args.raw_file, args.processed_file, args.knwl_ref_file, args.resp_ref_file)

    elif args.func == "get_knwl_gen_prompts":
        prompt_selection_for_knowledge_generation(
            args.test_file, args.train_file, args.model_file, 
            args.processed_file, args.data_type)
    
    elif args.func == "get_resp_gen_prompts":
        prompt_selection_for_response_generation(
            args.train_file, args.processed_file, args.seed)

    elif args.func == "prepare_input":
        prepare_input_for_response_generation(
            args.test_file, args.knwl_gen_file, args.processed_file)
