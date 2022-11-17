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


from galai.utils import escape_custom_split_sequence



class FastGalai(FastInferenceInterface):
    def __init__(self, model_name: str, args=None) -> None:
        super().__init__(model_name, args if args is not None else {})
        self.model = galai.load_model("huge")
        # warm up:
        output = self.model.generate("hello world", max_length=8)
        
        
    def dispatch_request(self, args, env) -> Dict:
        prompt = args[0]["prompt"]
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        max_tokens = args[0]["max_tokens"]
        top_p = args[0].get('top_p', 0)
        input_length = len(self.model.tokenizer.encode_batch([escape_custom_split_sequence(prompt)])[0].tokens)
        # print(input_length)
        output = self.model.generate(prompt, max_length=
                                     input_length+max_tokens, top_p=top_p)
        output = output.replace(prompt, "")
        choices = {"text":output}
        result={
            "result_type": RequestTypeLanguageModelInference,
            "choices": [choices],
        }
        print(f"<FastGalai.dispatch_request> return{result}")
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
