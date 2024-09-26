import torch.nn as nn


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


class LrWarmUp(object):
    def __init__(self, optimizer, min_rate=0.1, lr_decay=0.9, warm_steps=6000, reduce_steps=3000):
        self.optimizer = optimizer
        self.warm_steps = warm_steps
        self.reduce_steps = reduce_steps
        self.min_rate = min_rate
        self.lr_decay = lr_decay
        self.steps = 0
        self.new_steps = 0
        self.init_lrs = [param['lr'] for param in self.optimizer.param_groups]

    def step(self):
        self.steps += 1
        if self.steps <= self.warm_steps:
            for lr, param in zip(self.init_lrs, self.optimizer.param_groups):
                param['lr'] = lr * (self.steps / float(self.warm_steps))
        elif self.steps % self.reduce_steps == 0:
            for lr, param in zip(self.init_lrs, self.optimizer.param_groups):
                param['lr'] = max(lr * self.min_rate, param['lr'] * self.lr_decay)

    def back_step(self):
        pass

    def get_lr(self):
        return [param['lr'] for param in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'warm_steps': self.warm_steps,
            'reduce_steps': self.reduce_steps,
            'min_rate': self.min_rate,
            'lr_decay': self.lr_decay,
            'steps': self.steps,
            'new_steps': self.new_steps,
            'init_lrs': self.init_lrs
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.warm_steps = state_dict['warm_steps']
        self.reduce_steps = state_dict['reduce_steps']
        self.min_rate = state_dict['min_rate']
        self.lr_decay = state_dict['lr_decay']
        self.steps = state_dict['steps']
        self.new_steps = state_dict['new_steps']
        self.init_lrs = state_dict['init_lrs']
