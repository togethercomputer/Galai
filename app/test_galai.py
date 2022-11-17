import asyncio
import sys
from dacite import from_dict
import timeit
sys.path.append("./")

from common.together_web3.computer import LanguageModelInferenceRequest
from common.together_web3.together import TogetherWeb3

async def test():
    together_web3 = TogetherWeb3()
    with open("./foo_input.txt", 'r') as fp:
        line = fp.readline()
    result = await together_web3.language_model_inference(
        from_dict(
            data_class=LanguageModelInferenceRequest,
            data={
                "model": "galai",
                "max_tokens": 1024,
                "prompt": line,
            }
        ),
    )
    print("result", result)

if __name__=="__main__":
    # measure the end-to-end time
    start = timeit.default_timer()
    asyncio.run(test())
    end = timeit.default_timer()
    print("measure time: {}s".format( (end-start) ))