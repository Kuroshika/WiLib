import torch

def load_from_pretrained_model(model, block, args):
    def rm_module(old_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict

    if args.pretrained_model is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pre_trained_model)  # ['state_dict']
        if type(pretrained_dict) is dict and ('optimizer' in pretrained_dict.keys()):
            optimizer_dict = pretrained_dict['optimizer']
            pretrained_dict = pretrained_dict['model']
        pretrained_dict = rm_module(pretrained_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        keys = list(pretrained_dict.keys())
        for key in keys:
            for weight in args.ignore_weights:
                if weight in key:
                    if pretrained_dict.pop(key) is not None:
                        block.log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        block.log('Can Not Remove Weights: {}.'.format(key))
        block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
        model_dict.update(pretrained_dict)
        # block.log(model_dict)
        model.load_state_dict(model_dict)
        block.log('Pretrained model load finished: ' + args.pre_trained_model)
