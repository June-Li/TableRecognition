from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy

__all__ = ['build_optimizer']


def build_optimizer(config, epochs, step_each_epoch, parameters):
    from . import optimizer
    config = copy.deepcopy(config)
    # # step1 build lr
    lr = config.pop('lr')['learning_rate']

    # step2 build optimizer
    optim_name = config.pop('name')
    optim = getattr(optimizer, optim_name)(learning_rate=lr, **config)
    return optim(parameters), lr
