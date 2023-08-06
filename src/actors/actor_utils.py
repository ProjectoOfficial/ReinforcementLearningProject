from torch import nn

def build_sequential_network(dimensions:list, 
                             activation:nn.Module, 
                             last_activation:nn.Module, 
                             reset_parameters:bool=True, 
                             dropout_p:float=0.1, 
                             batch_norm:bool=False):
    layers = []
    for i in range(1, len(dimensions)):
        layer = nn.Linear(dimensions[i - 1], dimensions[i])
        
        if reset_parameters:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        if batch_norm and i > 1 and i < len(dimensions) - 1:
            layers.append(nn.BatchNorm1d(dimensions[i - 1]))
           
        layers.append(layer)
        
        if i < len(dimensions) - 1:
            layers.append(activation)
        else:
            if last_activation is not None:
                layers.append(last_activation)
        
        if dropout_p > 0 and i < len(dimensions) - 1:
            layers.append(nn.Dropout(dropout_p))
            
    return nn.Sequential(*layers)


def get_act_from_string(act: str):
    if act is None:
        return None
    elif act.lower() == "softmax":
        return nn.Softmax(dim=-1)
    elif act.lower() == "tanh":
        return nn.Tanh()
    elif act.lower() == "relu":
        return nn.ReLU()
    elif act.lower() == "leakyrelu":
        return nn.LeakyReLU()
    elif act.lower() == "hardtanh":
        return nn.Hardtanh()
    elif act.lower() == "hardsigmoid":
        return nn.Hardsigmoid()
    elif act.lower() == "logsigmoid":
        return nn.LogSigmoid()
    elif act.lower() == "hardswish":
        return nn.Hardswish()
    elif act.lower() == "relu6":
        return nn.ReLU6()
    elif act.lower() == "hardshrink":
        return nn.Hardshrink()
    else:
        raise NotImplementedError(f"add the activation function in actors/actor_utils.py")

