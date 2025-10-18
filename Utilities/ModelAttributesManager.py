class ModelAttributesManager:
    def ComputeParameters(model, text=False):
        tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
        if text:
            if tot >= 1e6:
                return '{:.2f}M'.format(tot / 1e6)
            else:
                return '{:.2f}K'.format(tot / 1e3)
        else:
            return tot


    def CreateOptimizer(param_list, optimizer_spec, load_sd=False):
        Optimizer = {
            'sgd': SGD,
            'adam': Adam
        }[optimizer_spec['name']]
        optimizer = Optimizer(param_list, **optimizer_spec['args'])
        if load_sd:
            optimizer.load_state_dict(optimizer_spec['sd'])
        return optimizer