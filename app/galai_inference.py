import galai
import asyncio
import base64
import json
import os
import sys
import time
from io import BytesIO
from random import randint
from typing import Dict
import requests
import re
sys.path.append("./")
from common.fast_inference import FastInferenceInterface
from common.together_web3.computer import RequestTypeLanguageModelInference
from common.together_web3.together import TogetherWeb3, TogetherClientOptions
from loguru import logger
import torch
from galai.utils import escape_custom_split_sequence



class FastGalai(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        self.seq_length_limit = 256
        print(f"<FastGalai.__init__> starts")
        galai_model = galai.load_model("huge")
        self.model = galai_model.model
        self.tokenizer = galai_model.tokenizer
        # warm up:
        print(f"<FastGalai.__init__> model loaded")
        print(f"<FastGalai.__init__> model: {self.model.config}")
        self.tokenizer.enable_truncation(max_length=self.seq_length_limit//2, direction="left")
        output = galai_model.generate("hello world", max_length=8)
        print(f"<FastGalai.__init__> model warmed")

    def dispatch_request(self, args, env) -> Dict:
        prompt = args[0]["prompt"]
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        max_tokens = args[0]["max_tokens"]
        top_p = args[0].get('top_p', 0)
        
        print(f"<dispatch_request> raw input seq <<{prompt}>>")
        encoded_ids = self.tokenizer.encode_batch([escape_custom_split_sequence(prompt)])[0].ids
        encoded_ids = torch.LongTensor([encoded_ids]).to(self.model.device)
        # print(f"<dispatch_request> encoded_input_ids shape {encoded_ids.shape} <<{encoded_ids}>")
        input_length = encoded_ids.shape[1]
        
        do_inference = True
        result_tokens = []
        while do_inference:
            print(f"<dispatch_request> input_length:{input_length}, max_tokens: {max_tokens}, seq_length_limit: {self.seq_length_limit}")
            output_pos = input_length
            if input_length + max_tokens <= self.seq_length_limit:
                max_length = input_length + max_tokens
                do_inference = False
            else:
                max_length = self.seq_length_limit
                max_tokens -= (self.seq_length_limit-input_length)
                input_length = self.seq_length_limit//2
                
            # print(f"do_inference: {do_inference}")
            if top_p is not None:
                out = self.model.generate(
                    encoded_ids, 
                    max_length=max_length, 
                    # min_length=max_length,
                    return_dict_in_generate=True, 
                    output_hidden_states=False,
                    top_p=top_p,
                    do_sample=True
                )
            else:
                out = self.model.generate(
                    encoded_ids, 
                    max_length=max_length, 
                    # min_length=max_length,
                    return_dict_in_generate=True, 
                    output_hidden_states=False
                )
            encoded_ids = out["sequences"][:,output_pos:]
            result_tokens.append(encoded_ids)
            print(f"<FastGalai.dispatch_request> Currrent encoded_ids shape <{encoded_ids.shape}>")
        
        # print(f"result_tokens: {result_tokens}")
        final_output = self.tokenizer.decode_batch(
            torch.cat(result_tokens, dim=1).tolist(), 
            skip_special_tokens=False)[0].lstrip('<pad>')
        choices = {"text":final_output}
        result={
            "result_type": RequestTypeLanguageModelInference,
            "choices": [choices],
        }
        print(f"<FastGalai.dispatch_request> return: {result}")
        return result

if __name__ == "__main__":
    coord_url = os.environ.get("COORD_URL", "127.0.0.1")
    coordinator = TogetherWeb3(
        TogetherClientOptions(),
        http_url=f"http://{coord_url}:8092",
        websocket_url=f"ws://{coord_url}:8093/websocket"
    )
    fip = FastGalai(model_name="galai", args={
        "coordinator": coordinator,
    })
    fip.start()
