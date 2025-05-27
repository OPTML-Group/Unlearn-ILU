import json

import torch
from transformers import Trainer

from .base import BaseTrainer
from .KL import kl_loss


class CUT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_id = kwargs.get("layer_id", 6)
        self.module_str = "{model_name}.model.layers[{layer_id}]"
        self.frozen_module = self.module_str.format(
            model_name="frozen_model", layer_id=self.layer_id
        )
        self.updated_module = self.module_str.format(
            model_name="updated_model", layer_id=self.layer_id
        )

    def get_steering_vec_list(self):
        with open("files/data/key_word_list/keywords.json") as f:
            keywords_list = json.load(f)
        steering_vectors_list = [[] for _ in range(len(keywords_list))]
        for i, keywords in enumerate(keywords_list):
            for keyword in keywords:
                steering_vectors_list[i].append(
                    self.get_steering_vec(
                        self.infer_model,
                        self.tokenizer,
                        keyword,
                        self.frozen_module,
                    )
                )

    def get_steering_vec(self, model, keyword, module):
        p_notice = f"You are a novice in {keyword} who often makes mistakes."
        p_expert = f"You are a world-class expert in {keyword}."
        inputs = self.tokenizer(
            [p_notice, p_expert], return_tensors="pt", padding=True, truncation=True
        ).cuda()
        activations = self.forward_with_cache(model, inputs, module, no_grad=True)
        direction = activations[0:1, -1:, :] - activations[1:, -1:, :]
        direction = direction.cuda()
        direction = direction / direction.norm(dim=-1, keepdim=True)

        return direction

    def forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []

        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None

        hook_handle = module.register_forward_hook(hook)

        if no_grad:
            with torch.no_grad():
                _ = model(**inputs)
        else:
            _ = model(**inputs)
        hook_handle.remove()

        return cache[0]
