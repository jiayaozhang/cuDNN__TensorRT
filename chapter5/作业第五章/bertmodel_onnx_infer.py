# Python 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)] on win32
# Type "help", "copyright", "credits" or "license()" for more information.
import torch
from torch.nn import functional as F
import numpy as np
import os
from transformers import BertTokenizer, BertForMaskedLM
import time

import onnx
import onnxruntime as ort
import transformers

print("pytorch:", torch.__version__)
print("onnxruntime version:", ort.__version__)
print("onnxruntime device:", ort.get_device())
print("transformers:", transformers.__version__)

BERT_PATH = 'bert-base-uncased'

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    if not os.path.exists(BERT_PATH):
        print(f"Download {BERT_PATH} model first!")
        assert(0)

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    # model = BertForMaskedLM.from_pretrained(BERT_PATH, return_dict = True)
    text = "The capital of France, " + \
        tokenizer.mask_token + ", contains the Eiffel Tower."
    encoded_input = tokenizer.encode_plus(text, return_tensors="pt")

    onnx_model = onnx.load(BERT_PATH+"/model.onnx")
    onnx.checker.check_model(onnx_model)

    # ort_session = ort.InferenceSession(BERT_PATH+"/model.onnx", providers=[
    #                                    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_session = ort.InferenceSession(
        BERT_PATH+"/model.onnx", providers=['CUDAExecutionProvider'])

    for n in ort_session.get_inputs():
        print(n.name)

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_inputs = {name: to_numpy(tensor) for name, tensor in encoded_input.items()}
    # print(ort_inputs)

    t1 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    print(f"*xiaofeng* Tort: {time.time() - t1}")
    # print(ort_outs)

    mask_index = torch.where(
            encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    out_tensor = torch.tensor(np.array(ort_outs[0]))
    print(out_tensor.shape)
    softmax = F.softmax(out_tensor, dim=-1)
    mask_word = softmax[0, mask_index, :]
    top_10 = torch.topk(mask_word, 10, dim=1)[1][0]
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(new_sentence)

    # model_test(model, tokenizer, text)
