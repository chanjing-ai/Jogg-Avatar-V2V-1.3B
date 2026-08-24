# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
TeaCache / CustomCache：与 wan2.2-5b-v2v-inference-main 一致的特征缓存，用于替代稀疏 CFG。
"""
import numpy as np
import torch


class TeaCache:
    """Timestep Embedding Aware Cache. 当累计相对 L1 距离未超过阈值则复用缓存，达到阈值则完整计算并清零."""

    def __init__(self, num_inference_steps, rel_l1_thresh, model_id, use_residual_caching=True):
        self.num_inference_steps = num_inference_steps
        self.rel_l1_thresh = rel_l1_thresh
        self.model_id = model_id
        self.use_residual_caching = use_residual_caching
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
            "Wan2.2-TI2V-5B": [1.57472669e+05, -1.15702395e+05, 3.10761669e+04, -3.83116651e+03, 2.21608777e+02, -4.81179567e+00],
            "infinitetalk-480": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],  # 1.3B
            "infinitetalk-720": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Choose from: {list(self.coefficients_dict)}")
        self.coefficients = self.coefficients_dict[model_id]
        self.use_raw_rel_change = model_id in ["Wan2.2-TI2V-5B", "infinitetalk-480", "infinitetalk-720"]
        self.reset()

    def reset(self):
        self.step = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit, x, t_mod):
        """
        返回 True 表示使用缓存（跳过完整计算），False 表示需要完整计算.
        """
        modulated_inp = t_mod.detach()
        if self.step == 0:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            diff = (modulated_inp - self.previous_modulated_input).abs().mean()
            base = self.previous_modulated_input.abs().mean() + 1e-8
            rel_change = (diff / base).cpu().item()
            if self.use_raw_rel_change:
                delta = rel_change
            else:
                rescale_func = np.poly1d(self.coefficients)
                delta = float(rescale_func(rel_change))
                delta = max(0.0, delta)
            self.accumulated_rel_l1_distance += delta
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_inp.clone()
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc and self.use_residual_caching:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        if self.use_residual_caching and self.previous_hidden_states is not None:
            self.previous_residual = hidden_states - self.previous_hidden_states
            self.previous_hidden_states = None
        else:
            self.previous_hidden_states = hidden_states.clone()

    def update(self, hidden_states):
        if self.use_residual_caching and self.previous_residual is not None:
            return hidden_states + self.previous_residual
        if self.previous_hidden_states is not None:
            return self.previous_hidden_states.clone()
        return hidden_states


class CustomCache(TeaCache):
    """
    CustomCache = TeaCache 决策 + TaylorSeer 式泰勒补偿（见 LightX2V cache_source.md）。
    """
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id, **kwargs):
        super().__init__(num_inference_steps, rel_l1_thresh, model_id, use_residual_caching=True)
        self._last_full_step = None
        self._taylor_cache = {}
        self._previous_residual = None

    def reset(self):
        super().reset()
        self._last_full_step = None
        self._taylor_cache = {}
        self._previous_residual = None

    def store(self, hidden_states):
        if self.previous_hidden_states is None:
            return
        current_residual = hidden_states - self.previous_hidden_states
        current_step = self.step - 1
        if self._previous_residual is not None and self._last_full_step is not None:
            step_diff_used = max(1, current_step - self._last_full_step)
            derivative = (current_residual - self._previous_residual.to(current_residual.device)) / step_diff_used
            self._taylor_cache = {0: current_residual.clone(), 1: derivative.clone()}
        else:
            self._taylor_cache = {0: current_residual.clone()}
        self._previous_residual = current_residual.clone()
        self._last_full_step = current_step
        self.previous_hidden_states = None

    def update(self, hidden_states):
        if not self._taylor_cache:
            return hidden_states
        step_diff = max(0, (self.step - 1) - (self._last_full_step or 0))
        residual_0 = self._taylor_cache[0]
        if residual_0.device != hidden_states.device:
            residual_0 = residual_0.to(hidden_states.device)
        out = residual_0
        if 1 in self._taylor_cache:
            residual_1 = self._taylor_cache[1].to(hidden_states.device)
            out = out + residual_1 * step_diff
        return hidden_states + out
