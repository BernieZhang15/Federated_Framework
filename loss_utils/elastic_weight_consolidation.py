import torch
import torch.nn.functional as F
from torch import autograd


class ElasticWeightConsolidation:

    def __init__(self, model,  weight=1000000):
        self.model = model
        self.weight = weight

    def update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.model.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def update_fisher_params(self, data_loader, num_batch):
        log_likelihoods = []
        for i, (input, target) in enumerate(data_loader):
            if i > num_batch:
                break
            output = F.log_softmax(self.model(input), dim=1)
            log_likelihoods.append(output[:, target])
        log_likelihood = torch.cat(log_likelihoods).mean()
        grad_log_likelihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_likelihood):
            self.model.register_buffer(_buff_param_name+'_estimated_fisher', param.data.clone() ** 2)

    def register_ewc_params(self, dataset, num_batches):
        self.update_fisher_params(dataset, num_batches)
        self.update_mean_params()

    def compute_consolidation_loss(self):
        try:
            losses = []
            for param_name, param in self.model.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.model, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.model, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (self.weight / 2) * sum(losses)
        except AttributeError:
            return 0


