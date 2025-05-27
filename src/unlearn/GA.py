import torch
from transformers import Trainer

from .base import BaseTrainer
from .KL import kl_loss


class GA(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        outputs = model(**forget_inputs)

        loss = -outputs.loss

        return (loss, outputs) if return_outputs else loss


class GA_FT(GA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]

        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        loss = -forget_outputs.loss + self.gamma * retain_outputs.loss
        return (loss, forget_outputs) if return_outputs else loss


class GA_KL(GA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]

        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        forget_outputs = model(**forget_inputs)
        retain_outputs = model(**retain_inputs)

        with torch.no_grad():
            infer_retain_outputs = self.infer_model(**retain_inputs)

        prob_retain_p = torch.softmax(retain_outputs.logits, dim=-1)
        prob_retain_q = torch.softmax(infer_retain_outputs.logits, dim=-1)

        forget_loss = forget_outputs.loss
        retain_loss = kl_loss(prob_retain_p, prob_retain_q)

        loss = -self.gamma * forget_loss + (1 - self.gamma) * retain_loss

        return (loss, forget_outputs) if return_outputs else loss

class NPO_FT(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }

        retain_data = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }

        outputs = model(**forget_inputs)
        current_forget_loss = outputs.loss

        with torch.no_grad():
            ref_outputs = self.infer_model(**forget_inputs)
            ref_forget_loss = ref_outputs.loss
        
        neg_log_ratios = current_forget_loss - ref_forget_loss

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        
        forget_loss = -torch.nn.functional.logsigmoid(0.1*neg_log_ratios).mean()*2/0.1

        loss = forget_loss + self.gamma * retain_loss

        return (loss, outputs) if return_outputs else loss

           