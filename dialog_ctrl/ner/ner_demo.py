
import torch
import numpy as np
from transformers import AutoTokenizer
from tabulate import tabulate

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
ner_model = torch.load("/gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/ner_model/roberta-large.pt")["model"]
ner_model.cuda()
ner_model.eval()

label_set = ["O", "B", "I"]

for step in range(100):
    print("===========================================================================")
    input_sent = input(">> Input:")
    tokens = input_sent.split()
    token_ids, first_tok_masks = [tokenizer.cls_token_id], [0]
    for token in tokens:
        subs_ = tokenizer.tokenize(token)
        assert len(subs_) > 0
        
        token_ids.extend(tokenizer.convert_tokens_to_ids(subs_))
        first_tok_masks.extend([1] + [0] * (len(subs_) - 1))
    
    token_ids.append(tokenizer.sep_token_id)
    first_tok_masks.append(0)
    
    token_ids = torch.LongTensor([token_ids]).cuda()
    predictions = ner_model(token_ids)  # (1, seq_len, 3)

    predictions = predictions[0].data.cpu().numpy() # (seq_len, 3)
    pred_ids = list(np.argmax(predictions, axis=1))

    assert len(pred_ids) == len(first_tok_masks)
    preds_for_each_word = []
    for pred, mask in zip(pred_ids, first_tok_masks):
        if mask == 1:
            preds_for_each_word.append(label_set[pred])

    assert len(preds_for_each_word) == len(tokens)
    table = [tokens, preds_for_each_word]
    print(tabulate(table))

    
