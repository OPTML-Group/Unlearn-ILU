import os
import sys
from copy import deepcopy
from time import time

sys.path.append("src")
import datasets
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer import is_datasets_available
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from pruner.utils import WrappedGPT, find_layers


class GenerateMask(Trainer):
    def __init__(self, score_type, ratios, mask_dir, p, q,mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_type = score_type
        self.ratios = [0.1,0.2,0.4,0.6,0.8,0.85,0.9,0.95,0.99,1.0]
        self.mask_dir = mask_dir
        self.p = p
        self.q = q
        self.mu = mu

    def score2mask(self, scores, ratio, return_rank=False):
        sorted_dict_positions = {}
        hard_dict = {}

        threshold_idx = int(len(scores) * ratio)
        positions = torch.argsort(scores)
        ranks = torch.argsort(positions)
        if return_rank:
            return ranks
        start_index = 0
        for key, tensor in self.model.named_parameters():
            num_elements = tensor.numel()
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_idx] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements
        return hard_dict

    def get_mask(self):
        if self.score_type == "gradient":
            self.gradient()
        elif self.score_type == "gradient_vis":
            self.gradient()
            torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
            exit(0)
        elif self.score_type == "weight":
            self.weight()
        elif self.score_type == "weight_vis":
            self.weight()
            torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
            exit(0)
        elif self.score_type == "random":
            self.random()
        elif self.score_type == "snip_advanced":
            self.snip_advanced()
        elif self.score_type == "snip_advanced_CL":
            self.snip_advanced(CL=True)
        elif self.score_type == "snip_advanced_gn":
            self.snip_advanced_gn()
        elif self.score_type == "snip_advanced_visualization":
            self.snip_advanced_visualization()
        elif self.score_type == "snip_advanced_new":
            self.snip_advanced_new()
        elif self.score_type == "FFN":
            hard_dict = {}
            for named, tensor in self.model.named_parameters():
                if "fc" in named or "final_layer_norm" in named:
                    hard_dict[named] = torch.ones_like(tensor)
                else:
                    hard_dict[named] = torch.zeros_like(tensor)
            torch.save(hard_dict, os.path.join(self.mask_dir, f"with_0.0.pt"))
            return
        elif self.score_type == "wanda":
            self.wanda()
            return
        else:
            raise ValueError(f"score_type {self.score_type} not supported")
        
        # Save the mask
        
        positions = torch.argsort(self.scores)
        ranks = torch.argsort(positions) # Get the rank of each element from low to high
        
        for ratio in self.ratios:
            if os.path.exists(os.path.join(self.mask_dir, f"with_{ratio}.pt")):
                continue
            sorted_dict_positions = {}
            hard_dict = {}

            threshold_idx = int(len(self.scores) * ratio)
            start_index = 0
            for key, tensor in self.model.named_parameters():
                num_elements = tensor.numel()
                # tensor_positions = positions[start_index: start_index + num_elements]
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_idx] = 1 # only keep the top k (minimum) elements
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements
            for key in hard_dict.keys():
                hard_dict[key] = hard_dict[key].type(torch.bool)
            torch.save(hard_dict, os.path.join(self.mask_dir, f"with_{ratio}.pt"))

    def gradient(self, dataset="forget"):
        gradients = {}

        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {dataset} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, dataset)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in gradients:
                        gradients[key] = tensor.grad.detach().clone()
                    else:
                        gradients[key] += tensor.grad.detach().clone()

            model.zero_grad()

        with torch.no_grad():
            for key, tensor in model.named_parameters():
                gradients[key] = -torch.abs(gradients[key])

            self.scores = torch.cat(
                [grad.flatten().cpu() for grad in gradients.values()]
            )

    def weight(self):
        begin_time = time()
        weights = {}
        with torch.no_grad():
            for key, tensor in self.model.named_parameters():
                weights[key] = -torch.abs(tensor.data)
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat([weight.flatten().cpu() for weight in weights.values()])

    def random(self):
        begin_time = time()
        random = {}
        with torch.no_grad():
            for key, tensor in self.model.named_parameters():
                random[key] = torch.rand_like(tensor.data)
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat(
                [random.flatten().cpu() for random in random.values()]
            )

    def hessianfree(self, retain_epoch=0, CL=False):
        hessianfree = {}
        retain_grad = {}
        for key, tensor in self.model.named_parameters():
            hessianfree[key] = 0
            retain_grad[key] = 0
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessianfree:
                        hessianfree[key] = tensor.grad.data
                    else:
                        hessianfree[key] += tensor.grad.data

            model.zero_grad()

        model.zero_grad()
        if retain_epoch > 0:
            epochs = retain_epoch
        else:
            epochs = 1
        for _ in range(epochs):
            for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "retain")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in retain_grad:
                            retain_grad[key] = tensor.grad.data
                        else:
                            retain_grad[key] += tensor.grad.data
                model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                hessianfree[key] = -hessianfree[key] * retain_grad[key]

            self.scores = torch.cat(
                [hessianfree.flatten().cpu() for hessianfree in hessianfree.values()]
            )

    def hessianfree_smooth(self, T=10, sigma=1e-5):
        hessianfree = {}
        retain_grad = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            hessianfree = {}
            retain_grad = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]
            for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "forget")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in hessianfree:
                            hessianfree[key] = tensor.grad.data
                        else:
                            hessianfree[key] += tensor.grad.data

                model.zero_grad()
            for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "retain")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in retain_grad:
                            retain_grad[key] = tensor.grad.data
                        else:
                            retain_grad[key] += tensor.grad.data
                model.zero_grad()
            with torch.no_grad():
                if t == 0:
                    for key, tensor in model.named_parameters():
                        hessianfree[key] = -torch.abs(hessianfree[key] * retain_grad[key]) / T
                else:
                    for key, tensor in model.named_parameters():
                        hessianfree[key] += -torch.abs(hessianfree[key] * retain_grad[key]) / T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
        with torch.no_grad():
            self.scores = torch.cat(
                [hessianfree.flatten().cpu() for hessianfree in hessianfree.values()]
            )


    def gradient_smooth(self, T=10, sigma=1e-5):
        scores = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            gradient = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]
            for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "forget")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in gradient:
                            gradient[key] = tensor.grad.data
                        else:
                            gradient[key] += tensor.grad.data

                model.zero_grad()
            if t == 0:
                for key, tensor in model.named_parameters():
                    scores[key] = -torch.abs(gradient[key]) / T
            else:
                for key, tensor in model.named_parameters():
                    scores[key] += -torch.abs(gradient[key]) / T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
        with torch.no_grad():
            self.scores = torch.cat(
                [gradient.flatten().cpu() for gradient in gradient.values()]
            )

    def hessian(self):
        mu = 1e-3
        forget_gradient = {}
        snip_hessian = {}
        retain_gradient_all = {}
        hessian = {}
        beta = 0.95
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradient:
                        forget_gradient[key] = tensor.grad.data
                    else:
                        forget_gradient[key] += tensor.grad.data

            model.zero_grad()
        for i in range(10):
            retain_gradient = {}
            for inputs in tqdm.tqdm(train_dataloader, desc="computing hessian"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "retain")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in retain_gradient:
                            retain_gradient[key] = tensor.grad.data
                        else:
                            retain_gradient[key] += tensor.grad.data
                        if key not in retain_gradient_all:
                            retain_gradient_all[key] = tensor.grad.data
                        else:
                            retain_gradient_all[key] += tensor.grad.data
                model.zero_grad()
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessian:
                        hessian[key] = retain_gradient[key] * retain_gradient[key]
                    else:
                        hessian[key] = (
                            beta * hessian[key]
                            + (1 - beta) * retain_gradient[key] * retain_gradient[key]
                        )
            model.zero_grad()

        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip_hessian[key] = -torch.abs(
                    mu
                    * (tensor.data - retain_gradient_all[key] / hessian[key])
                    * forget_gradient[key]
                    - mu
                    * mu
                    * retain_gradient_all[key]
                    / hessian[key]
                    * forget_gradient[key]
                )

            self.scores = torch.cat(
                [snip_hessian.flatten().cpu() for snip_hessian in snip_hessian.values()]
            )

    def snip(self, name, CL = False, FT = False, layer_wise = False):
        gradient = {}
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name, CL=CL, FT=FT)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in gradient:
                        gradient[key] = tensor.grad.data
                    else:
                        gradient[key] += tensor.grad.data

            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip[key] = -torch.abs(gradient[key] * tensor.data)
        if layer_wise:
            for ratio in self.ratios:
                W_masks = {}
                for key, tensor in model.named_parameters():
                    k = int(ratio * tensor.numel())
                    flat_tensor = tensor.flatten()
                    # Get the top k values and their indices from the flattened tensor
                    _, top_k_indices = torch.topk(flat_tensor, k, largest=True)
                    # Create a boolean mask for the flattened tensor
                    flat_mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
                    # Set True in the positions of the top k elements
                    flat_mask[top_k_indices] = True
                    # Reshape the flat mask back to the original tensor shape
                    mask = flat_mask.reshape(tensor.shape)
                    W_masks[key] = mask
                torch.save(W_masks, os.path.join(self.mask_dir, f"with_{ratio}.pt"))
            return
        else:
            with torch.no_grad():
                scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
            return scores

    def snip_forget_reinit(self, name):
        gradient = {}
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in gradient:
                        gradient[key] = tensor.grad.data
                    else:
                        gradient[key] += tensor.grad.data

            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip[key] = -torch.abs(gradient[key] * tensor.data)
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        mask = self.score2mask(scores, self.p, return_rank=False)
        del scores
        total = 0
        non_zero = 0
        with torch.no_grad():
            for key,tensor in model.named_parameters():
                total += tensor.numel()
                non_zero += mask[key].sum().item()
                tensor.data = tensor.data * (1-mask[key].to(tensor.device)) + torch.randn_like(tensor) * mask[key].to(tensor.device)
        print(f"total non zero: {non_zero}")
        print(f"Sparsity: {1-non_zero/total}")
        return






    def snip_smooth(self, name,T=10, sigma=1e-5):
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            gradient = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]            
            for inputs in tqdm.tqdm(train_dataloader, desc=f"computing forget gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "forget")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in gradient:
                            gradient[key] = tensor.grad.data
                        else:
                            gradient[key] += tensor.grad.data

                model.zero_grad()
            with torch.no_grad():
                if t == 0:
                    for key, tensor in model.named_parameters():
                        snip[key] = -torch.abs(gradient[key] * tensor.data)/T
                else:
                    for key, tensor in model.named_parameters():
                        snip[key] += -torch.abs(gradient[key] * tensor.data)/T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
                    
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        return scores



    def snip_visualization(self, name):
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in snip:
                        snip[key] = -torch.abs(tensor.grad.data * tensor.data)
                    else:
                        snip[key] += -torch.abs(tensor.grad.data * tensor.data)

            model.zero_grad()
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        torch.save(scores, os.path.join(self.mask_dir, f"scores.pt"))
        exit(0)

    def snip_advanced(self, CL=False):
        begin_time = time()
        forget_gradint = {}
        retain_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradint:
                        forget_gradint[key] = tensor.grad.data
                    else:
                        forget_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradint:
                        retain_gradint[key] = tensor.grad.data
                    else:
                        retain_gradint[key] += tensor.grad.data
            model.zero_grad()

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (tensor.data - retain_gradint[key] / self.mu) * forget_gradint[key]
                )
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )


    def snip_advanced_gn(self, CL=False):
        forget_gradint = {}
        retain_gradint = {}
        hessian = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradint:
                        forget_gradint[key] = tensor.grad.data
                    else:
                        forget_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradint:
                        retain_gradint[key] = tensor.grad.data
                    else:
                        retain_gradint[key] += tensor.grad.data
            model.zero_grad()
        total_hessian = 0
        num_hessian = 0
        for inputs in tqdm.tqdm(train_dataloader, desc="computing hessian"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)
            cnt = 0
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessian:
                        hessian[key] = tensor.grad.data 
                    else:
                        hessian[key] += tensor.grad.data
                cnt += 1
            model.zero_grad()
        for key in hessian.keys():
            hessian[key] = hessian[key] * hessian[key]/(cnt*cnt)
            num_elements = hessian[key].numel()
            num_hessian += num_elements
            total_hessian += hessian[key].sum().item()
        print(f"mean hessian: {total_hessian/num_hessian}")            

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (tensor.data - retain_gradint[key] / hessian[key]) * forget_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )


    def snip_advanced_visualization(self):
        mu = 1e-3
        forget_gradint = {}
        retain_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradint:
                        forget_gradint[key] = tensor.grad.data
                    else:
                        forget_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradint:
                        retain_gradint[key] = tensor.grad.data
                    else:
                        retain_gradint[key] += tensor.grad.data
            model.zero_grad()

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    mu * (tensor.data - retain_gradint[key]) * forget_gradint[key]
                    - mu * mu * retain_gradint[key] * forget_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )
        torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
        exit(0)

    def snip_advanced_new(self):
        forget_gradint = {}
        retain_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in forget_gradint:
                        forget_gradint[key] = tensor.grad.data
                    else:
                        forget_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in retain_gradint:
                        retain_gradint[key] = tensor.grad.data
                    else:
                        retain_gradint[key] += tensor.grad.data
            model.zero_grad()
        mu = 1e-3
        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (
                        mu
                        - (mu / (tensor.data + 1e-12))
                        * (1 + mu / (tensor.data + 1e-12))
                        * retain_gradint[key]
                    )
                    * forget_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )

    def normalizedhf(self):
        hessianfree = {}
        retain_grad = {}
        for key, tensor in self.model.named_parameters():
            hessianfree[key] = 0
            retain_grad[key] = 0
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing forget gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "forget")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    hessianfree[key] += tensor.grad.data.cpu()

            model.zero_grad()

        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing retain gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "retain")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    retain_grad[key] += tensor.grad.data.cpu()
            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                hessianfree[key] = (
                    -hessianfree[key]
                    * retain_grad[key]
                    / (torch.abs(tensor.data.cpu()) + 1e-8)
                )
            self.scores = torch.cat(
                [hessianfree.flatten() for hessianfree in hessianfree.values()]
            )

    def wanda(self):
        begin_time = time()
        W_metrics = {}

        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        use_cache = model.config.use_cache
        model.config.use_cache = False

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = self.prepare_calibration_input(
                model, train_dataloader
            )
        try:
            layers = model.model.layers
        except:
            layers = model.model.decoder.layers
        cnt = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                )

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(len(train_dataloader)):
                with torch.no_grad():
                    if position_ids is not None:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
            for handle in handles:
                handle.remove()

            for name in subset:
                W_metrics[cnt] = (
                    (
                        torch.abs(subset[name].weight.data)
                        * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    )
                    .detach()
                    .cpu()
                )
                cnt += 1
            for j in range(len(train_dataloader)):
                with torch.no_grad():
                    if position_ids is not None:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
            inps, outs = outs, inps
        end_time = time()
        print(f"Mask generation Time taken: {end_time-begin_time}")
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        for ratio in self.ratios:
            W_masks = {}
            cnt = 0
            for i in range(len(layers)):
                layer = layers[i]
                subset = find_layers(layer)

                for name in subset:
                    W_metric = W_metrics[cnt]
                    W_mask = torch.zeros_like(W_metric) == 1
                    sort_res = torch.sort(W_metric, dim=-1, stable=True) # sort the weights into ascending order
                    indices = sort_res[1][:, : int(W_metric.shape[1] * (1-ratio))]
                    W_mask.scatter_(1, indices, True)
                    W_masks[cnt] = ~W_mask
                    cnt += 1
            for i in range(len(W_masks)):
                W_masks[i] = W_masks[i].type(torch.bool)
            torch.save(W_masks, os.path.join(self.mask_dir, f"with_{ratio}.pt"))

    def prepare_calibration_input(self, model, dataloader):
        use_cache = model.config.use_cache
        model.config.use_cache = False
        try:
            layers = model.model.layers
        except:
            layers = model.model.decoder.layers
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
        else:
            device = next(iter(model.parameters())).device
        dtype = next(iter(model.parameters())).dtype
        batch = next(iter(dataloader))
        batch = batch["forget"]
        inps = torch.zeros(
            (len(dataloader), batch[0].shape[1], model.config.hidden_size),
            dtype=dtype,
            device=device,
        )
        inps.requires_grad = False
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                try:
                    cache["position_ids"] = kwargs["position_ids"]
                except:
                    pass
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            batch = batch["forget"]
            try:
                model(input_ids=batch[0], attention_mask=batch[1])
            except ValueError:
                pass
        layers[0] = layers[0].module
        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, batch[0].shape[1]), dtype=torch.long, device=inps.device
            )

            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (1,  batch[0].shape[1]),
                inps,
                0,
            )
        try:
            position_ids = cache["position_ids"]
        except:
            position_ids = None
        model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids

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

    def compute_loss_adapted(self, model, inputs, key, CL=False, FT = False, return_outputs=False):
        data = inputs[key]
        retain_data = inputs["retain"]
        if CL and key == "forget":
            forget_data = data
            input_ids = forget_data[0].clone()
            labels = forget_data[3]
            postions = forget_data[4]
            pad_id = input_ids[0][-1].item()
            for idx, position in enumerate(postions):
                input_ids[idx, position:] = labels[idx][position:].clone()
                mask = input_ids[idx] == -100
                input_ids[idx, mask] = pad_id
            inputs = {
                "input_ids": input_ids,
                "attention_mask": forget_data[1],
                "labels": labels,
            }
        else:
            inputs = {
                "input_ids": data[0],
                "attention_mask": data[1],
                "labels": data[2],
            }

        outputs = model(**inputs)

        loss = outputs.loss
        if FT:
            retain_inputs = {
                "input_ids": retain_data[0],
                "attention_mask": retain_data[1],
                "labels": retain_data[2],
            }
            retain_outputs = model(**retain_inputs)
            loss += retain_outputs.loss
        return (loss, outputs) if return_outputs else loss
