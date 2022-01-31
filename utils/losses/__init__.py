import copy
from .table_att_loss import TableAttentionLoss


def build_loss(config):
    support_dict = ['TableAttentionLoss']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
