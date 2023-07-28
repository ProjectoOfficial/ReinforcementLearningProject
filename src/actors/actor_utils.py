from torch import nn

def build_sequential_network(dimensions:list, activation:nn.Module, last_activation:nn.Module, xavier_init:bool=True):
    layers = []
    for i in range(1, len(dimensions)):
        layer = nn.Linear(dimensions[i-1], dimensions[i])
        
        if xavier_init:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        layers.append(layer)
        
        if i < len(dimensions) - 1:
            layers.append(activation)
        else:
            if last_activation is not None:
                layers.append(last_activation)
            
    return nn.Sequential(*layers)
