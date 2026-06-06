import torch
import torch.nn as nn

class GradientMonitor:
    """记录各层梯度变化并动态选择稳定层"""
    def __init__(self, model, threshold=0.5, window_size=5):
        self.grad_history = {}  # 记录历史梯度（滑动窗口）
        self.threshold = threshold
        self.window_size = window_size
        self.hooks = []
        
        # 为每一层注册梯度钩子
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Module):  # 注册所有子模块的钩子
                self.hooks.append(
                    layer.register_backward_hook(self._hook_grad(name))
                )
                self.grad_history[name] = []
                
    
    def _hook_grad(self, name):
        def hook(module, grad_input, grad_output):
            grad_norm = grad_output[0].abs().mean().item()
            self.grad_history[name].append(grad_norm)
            # 保留最近window_size次梯度值
            if len(self.grad_history[name]) > self.window_size:
                self.grad_history[name].pop(0)
        return hook
    
    def get_stable_layers(self):
        stable_layers = []
        for name, grads in self.grad_history.items():
            if len(grads) < self.window_size:
                continue
            # 判断梯度是否趋于稳定（最近window_size次梯度均低于阈值）
            if all(g < self.threshold for g in grads[-self.window_size:]):
                stable_layers.append(name)
        return stable_layers