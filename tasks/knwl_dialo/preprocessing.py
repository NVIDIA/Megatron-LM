
"""Preprocessing for Wizard of Wikipedia and Wizard of Internet datasets"""

import argparse
from nltk import word_tokenize
from tqdm import tqdm
import numpy as np
import json

def get_params():
    parser = argparse.ArgumentParser(description="Preprocessing")

    parser.add_argument("--func", type=str, default="",
                        help="choose to run which function")
    parser.add_argument("--input_file", type=str, default="",
                        help="path of the input file")
    parser.add_argument("--knowledge_file", type=str, default="",
                        help="path of the knowledge file")
    parser.add_argument("--test_file", type=str, default="",
                        help="path of the test file")
    parser.add_argument("--train_file", type=str, default="",
                        help="path of the train file")
    parser.add_argument("--output_file", type=str, default="",
                        help="path of the output file")
    parser.add_argument("--model_file", type=str, default="",
                        help="path of the model file")
    parser.add_argument("--seed", type=int, default=123456,
                        help="random seed")

    params = parser.parse_args()
    return params


def process_wow_dataset(input_file, output_file):
    """
      This is a function used for processing the wizard of wikipedia (wow) dataset
      Expected processed format:
      topic \t dialogue context \t golden knowledge \t golden response
    """

    with open(input_file, "r") as fr:
        dialog_data = json.load(fr)
    
    with open(output_file, "w") as fw:
        for i, sample in enumerate(tqdm(dialog_data)):
            # get all the dialog data for a single sample
            dialog = sample["dialog"]
            
            context = []
            for j, turn in enumerate(dialog):
                text = turn["text"]
                if not (text.endswith("?") or text.endswith(".") or text.endswith("!")):
                    text = text + " ."
                text = " ".join(word_tokenize(text))
                
                if j == 0:
                    # first turn
                    context.append(text)
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
                    
                    # write to the output file
                    fw.write(topic + "\t" + " [SEP] ".join(context) + "\t" + \
                                checked_sentence + "\t" + text + "\n")
                    context.append(text)

                else:
                    assert "apprentice" in speaker
                    context.append(text)


def process_woi_dataset(input_file, output_file):
    """
      This is a function used for processing the wizard of internet (woi) dataset
      Expected processed format:
      topic \t dialogue context \t golden knowledge \t golden response
    """

    with open(output_path, "w") as fw:
        with open(input_path, "r") as fr:
            for i, line in tqdm(enumerate(fr)):
                line = line.strip()
                item_dict = json.loads(line)
                item_dict = item_dict.values()
                assert len(item_dict) == 1
                item_dict = list(item_dict)[0]
                
                dialog_data = item_dict['dialog_history']
                length = len(dialog_data)
                
                turn_list = []
                search_text = ""
                for i in range(length):
                    item = dialog_data[i]
                    action = item['action']

                    if action == "Wizard => SearchAgent":
                        search_text = item['text']

                    elif action == "Wizard => Apprentice":
                        if len(turn_list) == 0:
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
                            # no knowledge sentence is used
                            topic = "no_topic"
                            sent_list = ["no_passages_used"]
                        else:
                            # assert search_text != ""
                            topic = search_text

                            sent_list = []
                            for content, select in zip(contents, selects):
                                content = content['content']
                                assert len(content) == len(select)
                                for c, s in zip(content, select):
                                    if s:
                                        sent_list.append(c)
                        if len(sent_list) == 0:
                            topic = "no_topic"
                            sent_list = ["no_passages_used"]

                        # get dialogue context, knowledge, and response 
                        dialog_context = " [SEP] ".join(turn_list)
                        knwl_sent = sent_list[0]
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
                        
                        # write to the ouput file
                        if topic != "no_topic":
                            fw.write(topic + "\t" + dialog_context + "\t" + \
                                     knwl_sent + "\t" + response + "\n")

                        turn_list.append(response)

                    elif action == "Apprentice => Wizard":
                        turn = item['text']
                        turn_list.append(turn)

                    else:
                        assert action == "SearchAgent => Wizard"


def get_database(test_datapath, train_datapath):
    """Get the database sorted by topics"""

    # get test data topic list
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
            if knowledge == "no_passages_used":
                continue
            
            # get the instance
            last_turn = turns[-1]
            instance = "( " + last_turn + " ) " + topic + " => " + knowledge
            
            # construct dialog example
            dialog_example = ""
            dialog_example += "( " + topic + " )"
            for turn in turns:
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
        test_datapath, train_datapath, model_path, output_prompt_path):
    """Selecting prompts for the knowledge generation"""

    print("> Selecting prompts for the knowledge generation")

    train_data_by_topic, dialog_data_by_topic, dialog_examples = \
                            get_database(test_datapath, train_datapath)
    
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
    count_out_of_list = 0
    prompt_list_for_each_sample = []
    with open(test_datapath, "r") as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()

            splits = line.split("\t")
            topic = splits[0]
            turns = splits[1].split(" [SEP] ")[-3:]

            if topic not in train_data_by_topic:
                count_out_of_list += 1

                # calculate similarity
                # get the query embedding
                query_sent = ""
                query_sent += "( " + topic + " )"
                for turn in turns:
                    query_sent += " "
                    query_sent += turn
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
                # query_sent
                query_sent = ""
                query_sent += "( " + topic + " )"
                for turn in turns:
                    query_sent += " "
                    query_sent += turn

                dialog_list = dialog_data_by_topic[topic]
                assert len(dialog_list) == num_data_sample

                # calculate the similarity
                selected_examples = select_prompts_based_on_similarity(
                                query_sent, dialog_list, total_example_list, 
                                topic, tokenizer, encoder, topk=num_data_sample)
                example_list = selected_examples
                
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
            response_token_list = response.split()
            response_len = len(response_token_list)
            num_overlap_token = 0
            for token in response_token_list:
                if token in knowledge_sent_token_dict:
                    num_overlap_token += 1
            
            # filtering the data based on the ratio
            if num_overlap_token > response_len * 0.9 or num_overlap_token < response_len * 0.6:
                continue

            prompt_example = ""
            # add dialog context
            prompt_example += "Topic: " + topic + ". "
            prompt_example += "User says: " + turns[-1] + " "
            prompt_example += "We know that: " + knowledge + " "
            prompt_example += "System replies: " + response
            
            prompt_example_list.append(prompt_example)
        
    print("> shuffle the prompt examples (total %d)" % len(prompt_example_list))
    np.random.shuffle(prompt_example_list)

    print("> Prompt example:")
    print(prompt_example_list[0])
    
    print("> writing to %s" % output_path)
    with open(output_path, "w") as f:
        # f.write("Generate the System's response based on the knowledge sentence:\n")
        for i in tqdm(range(20)):
            example = prompt_example_list[i]
            f.write(example + "\n")


def prepare_input_for_response_generation(test_file, knowledge_file, output_file):
    """Preparing inputs for the response generation"""

    # get the knowledge list
    with open(knowledge_file, "r") as f:
        knowledge_list = f.readlines()
    
    with open(test_file, "r") as fr:
        with open(output_file, "w") as fw:
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

    params = get_params()
    if params.func == "process_wow_dataset":
        process_wow_dataset(params.input_file, params.output_file)

    elif params.func == "process_woi_dataset":
        process_woi_dataset(params.input_file, params.output_file)

    elif params.func == "get_prompts":
        prompt_selection_for_knowledge_generation(
            params.test_file, params.train_file, params.model_file, params.output_file)
        prompt_selection_for_response_generation(
            params.train_file, params.output_file, params.seed)

    elif params.func == "prepare_input":
        prepare_input_for_response_generation(
            params.test_file, params.knowledge_file, params.output_file)
