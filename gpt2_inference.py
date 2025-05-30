from transformers import pipeline, set_seed
import torch
import numpy as np
from custom_matmul_cpp import custom_matmul_cpp

# original_forward = torch.nn.Linear.forward

def custom_forward(self, input=None, bias=None):
    output_shape = input.shape[:-1] + (self.weight.shape[::-1][-1],)
    input = input.reshape(-1, input.shape[-1])
    output = custom_matmul_cpp(input, self.weight.t())
    # print("\033[1;33m" + "matmul called from python" + "\033[0m")
    # output = input @ self.weight.t() + (bias if bias is not None else 0)
    output = output.reshape(output_shape)
    return output

torch.nn.Linear.forward = custom_forward
set_seed(42)
generator = pipeline('text-generation', model='/Users/ashwinrohit/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e', device="mps")
print(generator("Hello, I'm a language model,", max_new_tokens=20))