import torch
import torch.nn as nn

import balorgnn.train.layers as l
    
# def f1_loss(y_pred, y_true, epsilon=1e-7):
#     # Convert predictions to binary (0 or 1) based on a threshold

    
#     # Calculate true positives, false positives, and false negatives
#     tp = (y_true * y_pred).sum().to(torch.float32)
#     fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
#     fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

#     # Calculate precision and recall
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)

#     # Calculate F1 score
#     f1 = 2 * (precision * recall) / (precision + recall + epsilon)

#     # F1 loss is 1 - F1 score
#     return 1 - f1

# def class_rmse_loss(y_pred, y_true, epsilon=1e-7):
#     # Convert predictions to binary (0 or 1) based on a threshold

    
    
#     # Calculate true positives, false positives, and false negatives
#     tp = (y_true * y_pred).sum().to(torch.float32)
#     fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
#     fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

#     # Calculate precision and recall
#     precision = tp / (tp + fp + epsilon)
#     recall = tp / (tp + fn + epsilon)

#     # Calculate F1 score
#     f1 = 2 * (precision * recall) / (precision + recall + epsilon)

#     # F1 loss is 1 - F1 score
#     return 1 - f1


class BaseArch(torch.nn.Module):
    def __init__(self, feature_size, outputs, hidden_channels):
        super(BaseArch, self).__init__()

        self.MLPs = nn.ModuleList()

        self.outputs = outputs

        self.sigmoid_valid = nn.Sigmoid()
        self.sigmoid_synth = nn.Sigmoid()
        self.sigmoid_oversized = nn.Sigmoid()

        for _ in range(len(outputs)):
            mlp = l.MLP(feature_size, 1, activation_type="elu",
                                    hidden_channels=hidden_channels,
                                    num_hidden_lyr=len(hidden_channels))
                            
            self.MLPs.append(mlp)


    def forward(self, data):
        embedding = data.x
        
        outs = []
        for layer in self.layers:
            embedding = layer.forward(data, embedding, outs)
            if layer.clear_outs:
                outs = []
            else:
                outs.append(embedding)


        total_loss = 0
        
        loss_dict = {}
        out_dict = {}

        for i, target in enumerate(self.outputs):
            ground_truth = data.y[:, i].reshape([-1, 1])
            out = self.MLPs[i](embedding)

            use_in_loss_mask = data.use_in_loss_mask[:, i]

            if use_in_loss_mask.any():
                ground_truth_temp = ground_truth[use_in_loss_mask]

                bce_loss = False

                if "Valid" in target:
                    out = self.sigmoid_valid(out)
                    bce_loss = True
                elif "Synthesized" in target:
                    out = self.sigmoid_synth(out)
                    bce_loss = True
                elif "Oversized" in target:
                    out = self.sigmoid_synth(out)
                    bce_loss = True

                out_temp = out[use_in_loss_mask]

                if not bce_loss:
                    loss = torch.sqrt(torch.nn.MSELoss()(out_temp, ground_truth_temp) + 0.001)
                else:
                    loss = nn.BCELoss()(out_temp, ground_truth_temp)
                    
                if torch.isnan(loss).any():
                    print(use_in_loss_mask)
                    print(out_temp, ground_truth_temp, loss)
                    quit()

                total_loss += loss
                loss_dict[(target, i)] = loss

            out_dict[(target, i)] = out

        return out_dict, total_loss, loss_dict
    
    

class CatArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs, output_config_name):
            feature_size = 64

            super(CatArch, self).__init__(feature_size, outputs, output_config_name, [32, 16, 8])
            layers = []

            layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))

            for _ in [1, 2, 3, 4, 5]:
                layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

            layers.append(l.JKN())
            layers.append(l.NodeToGraphAggregate(feature_size))

            self.layers = nn.ModuleList(layers)

class BullArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs):
            feature_size = 64

            super(BullArch, self).__init__(feature_size, outputs, [32, 16, 8])
            layers = []

            layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))
            layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

            layers.append(l.JKN())
            layers.append(l.NodeToGraphAggregate(feature_size))

            self.layers = nn.ModuleList(layers)

class RhinoArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs):
        feature_size = 64

        super(RhinoArch, self).__init__(feature_size, outputs, [32, 16, 8])
        layers = []

        layers.append(l.ResidualBlockLayer(input_size))
        layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))
        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.NodeToGraphAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))

        self.layers = nn.ModuleList(layers)


class SnakeArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs):
        feature_size = 64

        super(SnakeArch, self).__init__(feature_size, outputs, [32, 16, 8])
        layers = []

        layers.append(l.ResidualBlockLayer(input_size))
        layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))
        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.NodeToBasicBlockAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.BasicBlockTransformerConvLayer(feature_size, feature_size))

        layers.append(l.BasicBlockToGraphAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))

        self.layers = nn.ModuleList(layers)

class CamelArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs, output_config_name):
        feature_size = 64

        super(CamelArch, self).__init__(feature_size, outputs, output_config_name, [32, 16, 8])
        layers = []

        layers.append(l.ResidualBlockLayer(input_size))
        layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.NodeToBasicBlockAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.BasicBlockTransformerConvLayer(feature_size, feature_size))

        layers.append(l.BasicBlockToGraphAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))

        self.layers = nn.ModuleList(layers)



class DogArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs):
        feature_size = 128

        super(DogArch, self).__init__(feature_size, outputs, [64, 32, 16, 8])
        layers = []

        layers.append(l.ResidualBlockLayer(input_size))
        layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.NodeToBasicBlockAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.BasicBlockTransformerConvLayer(feature_size, feature_size))

        layers.append(l.BasicBlockToGraphAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))

        self.layers = nn.ModuleList(layers)



class MouseArch(BaseArch):
    def __init__(self, input_size, edge_dim, outputs):
        feature_size = 64

        super(MouseArch, self).__init__(feature_size, outputs, [32, 16, 8])
        layers = []

        layers.append(l.ResidualBlockLayer(input_size))
        layers.append(l.NodeTransformerConvLayer(input_size, feature_size, edge_dim))
        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.NodeTransformerConvLayer(feature_size, feature_size, edge_dim))

        layers.append(l.NodeToBasicBlockAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))
        layers.append(l.BasicBlockTransformerConvLayer(feature_size, feature_size))
        layers.append(l.ResidualBlockLayer(feature_size))

        layers.append(l.BasicBlockToGraphAggregate(feature_size))

        layers.append(l.ResidualBlockLayer(feature_size))

        self.layers = nn.ModuleList(layers)