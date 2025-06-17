from accelerate import Accelerator
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_community.llms import CTransformers
from accelerate import Accelerator
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import VLLM
from langchain_huggingface.llms import HuggingFacePipeline


allowed_llm_sources = ['google', 'gguf-llama-cpp', 'gguf-ctransformers' , 'transformers-pipeline', 'vllm']
llm_source = "transformers-pipeline"
# fill this part if llm has a gguf file
gguf_path = "models/gguf/llama-2-7b.Q4_K_M.gguf"
embedding_sources = ['google', 'huggingface']
embedding_source = embedding_sources[1]

import os

llm = None
if llm_source == 'google':
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=os.environ['gemini_api_key']
    )
elif llm_source == 'gguf-ctransformers':
    accelerator = Accelerator()
    config = {'max_new_tokens': 512, 'repetition_penalty': 1.1,
              'context_length': 8000, 'temperature': 0, 'gpu_layers': 10, 'stream': True}
    llm = CTransformers(model=gguf_path,
                        model_type="llama", stream=True, config=config)
    llm, config = accelerator.prepare(llm, config)
elif llm_source == 'gguf-llama-cpp':
    # Change this value based on your model and your GPU VRAM pool.
    n_gpu_layers = 160
    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_batch = 4096
    n_ctx = 4096

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=gguf_path,
        n_gpu_layers=n_gpu_layers, n_batch=n_batch,
        callback_manager=callback_manager,
        temperature=0.2,
        max_tokens=2000,
        top_p=1,
        verbose=False,
        n_ctx=n_ctx,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        model_kwargs={
            "repetition_penalty": 5
        }
    )

elif llm_source == 'transformers-pipeline':
   import transformers
   import torch
   import os
   # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
   model_id = "selfrag/selfrag_llama2_7b"
   hf_token = os.environ['hf_token']
#    bnb_config = transformers.BitsAndBytesConfig(
#       load_in_8bit=True,
#       bnb_8bit_compute_dtype=torch.bfloat16
#    )
   pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={
      "torch_dtype": torch.bfloat16,
      # "quantization_config": bnb_config
   }, token=hf_token, do_sample=True,
      max_new_tokens=200, top_p=1.0, temperature=0.01, return_full_text=False, device_map="cuda")
   llm = HuggingFacePipeline(pipeline=pipeline)
else:
   llm = VLLM(
    model="selfrag/selfrag_llama2_7b",
    max_num_seqs=1,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=200,
    top_p=1,
    temperature=0,
    # seed=42,
    dtype='half',
    vllm_kwargs={
        "seed": 42,
        "gpu_memory_utilization": 0.75,
        "max_num_seqs" : 1,
    },
   )