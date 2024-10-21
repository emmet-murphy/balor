
from torch.nn import Sequential, Linear, ReLU

from torch_scatter import scatter_add
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import reset

import torch

import torch.nn as nn
    
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge, TransformerConv

class BaseLayer(torch.nn.Module):
    def __init__(self):
        super(BaseLayer, self).__init__()
        self.clear_outs = False

class ClearOutsLayer(BaseLayer):
    def __init__(self):
        super(ClearOutsLayer, self).__init__()
        self.clear_outs = True

    def forward(self, data, input, outs):
        return input


class NodeTransformerConvLayer(BaseLayer):
    def __init__(self, in_size, out_size, edge_dim):
        super(NodeTransformerConvLayer, self).__init__()
        self.transformer_conv = TransformerConv(in_size, out_size, edge_dim=edge_dim)

    def forward(self, data, input, outs):
        out = self.transformer_conv(input, data.edge_index, edge_attr=data.edge_attr)
        return F.elu(out)
    

class BasicBlockTransformerConvLayer(BaseLayer):
    def __init__(self, in_size, out_size):
        super(BasicBlockTransformerConvLayer, self).__init__()
        self.transformer_conv = TransformerConv(in_size, out_size)

    def forward(self, data, input, outs):
        out = self.transformer_conv(input, data.cfg_edge_index)
        return F.elu(out)
    

class ResidualBlockLayer(BaseLayer):
    def __init__(self, size):
        super(ResidualBlockLayer, self).__init__()
        self.residual_block = ResBlock(size)

    def forward(self, data, input, outs):
        return self.residual_block(input)
    
class NodeToBasicBlockAggregate(BaseLayer):
    def __init__(self, size):
        super(NodeToBasicBlockAggregate, self).__init__()
        self.gate_nn = Sequential(Linear(size, size), ReLU(), Linear(size, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)
        self.clear_outs = True

    def forward(self, data, input, outs):
        out, _ = self.glob(input, data.bb_id_list)
        return out
    
class NodeToGraphAggregate(BaseLayer):
    def __init__(self, size):
        super(NodeToGraphAggregate, self).__init__()
        self.gate_nn = Sequential(Linear(size, size), ReLU(), Linear(size, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)
        self.clear_outs = True

    def forward(self, data, input, outs):
        out, _ =  self.glob(input, data.batch)
        return out

class BasicBlockToGraphAggregate(BaseLayer):
    def __init__(self, size):
        super(BasicBlockToGraphAggregate, self).__init__()
        self.gate_nn = Sequential(Linear(size, size), ReLU(), Linear(size, 1))
        self.glob = MyGlobalAttention(self.gate_nn, None)
        self.clear_outs = True

    def forward(self, data, input, outs):
        out, _ =  self.glob(input, data.bb_batch)
        outs = []
        return out
    
class JKN(BaseLayer):
    def __init__(self):
        super(JKN, self).__init__()
        self.jkn = JumpingKnowledge('max')
        self.clear_outs = True

    def forward(self, data, input, outs):
        return self.jkn(outs)
    


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Linear(num_features, num_features)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.conv2 = nn.Linear(num_features, num_features)
        self.bn2 = nn.BatchNorm1d(num_features)


    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out



class MyGlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super(MyGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        gate = self.gate_nn(x).view(-1, 1)
        
        x = self.nn(x) if self.nn is not None else x

        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    

class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]
    
def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    if act == 'elu' or act == 'elu+1':
        return nn.ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))
