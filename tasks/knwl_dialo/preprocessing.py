
import argparse
from nltk import word_tokenize
from tqdm import tqdm

def get_params():
    parser = argparse.ArgumentParser(description="Preprocessing")

    parser.add_argument("--func", type=str, default="")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--knowledge_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")

    params = parser.parse_args()
    return params


def process_wow_dataset(input_file, output_file):
    """
      expected processed format:
      topic \t dialogue context \t golden knowledge \t golden response
    """
    with open(input_file, "r") as fr:
        dialog_data = json.load(fr)
    
    with open(output_file, "w") as fw:
        for i, sample in enumerate(tqdm(dialog_data)):
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

                    if len(checked_sentence) > 0:
                        checked_sentence = checked_sentence[0]
                    else:
                        checked_sentence = "no_passages_used"

                    if len(checked_passage) == 1:
                        checked_passage = checked_passage[0]
                    else:
                        checked_passage = "no_passages_used"

                    if checked_passage != "no_passages_used":
                        topic = checked_passage
                    else:
                        topic = sample["chosen_topic"]
                    
                    fw.write(topic + "\t" + " [SEP] ".join(context) + "\t" + checked_sentence + "\t" + text + "\n")

                    context.append(text)

                else:
                    assert "apprentice" in speaker
                    context.append(text)


def process_woi_dataset(input_file, output_file):
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

                        # get knowledge sentence
                        contents = item["context"]["contents"]
                        selects = item["context"]["selected_contents"]
                        flag = selects[0][0]
                        selects = selects[1:]
                        assert len(selects) == len(contents)
                        
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
                        
                        dialog_context = " [SEP] ".join(turn_list)
                        knwl_sent = sent_list[0]
                        response = item['text']

                        topic = topic.replace("\n", "")
                        topic = topic.replace("\r", "")
                        topic = topic.replace("\t", "")
                        
                        dialog_context = dialog_context.replace("\n", "")
                        dialog_context = dialog_context.replace("\r", "")
                        dialog_context = dialog_context.replace("\t", "")

                        knwl_sent = knwl_sent.replace("\n", "")
                        knwl_sent = knwl_sent.replace("\r", "")
                        knwl_sent = knwl_sent.replace("\t", "")

                        response = response.replace("\n", "")
                        response = response.replace("\r", "")
                        response = response.replace("\t", "")
                        
                        if topic != "no_topic":
                            fw.write(topic + "\t" + dialog_context + "\t" + knwl_sent + "\t" + response + "\n")

                        turn_list.append(response)

                    elif action == "Apprentice => Wizard":
                        turn = item['text']
                        turn_list.append(turn)

                    else:
                        assert action == "SearchAgent => Wizard"



if __name__ == "__main__":

    params = get_params()
    if params.func == "process_wow_dataset":
        process_wow_dataset(params.input_file, params.output_file)

    elif params.func == "process_woi_dataset":
        process_woi_dataset(params.input_file, params.output_file)
