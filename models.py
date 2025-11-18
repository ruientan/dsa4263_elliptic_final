import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_classes=2, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GraphSAGE(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_classes=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        weights = self.class_weights[targets]
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')


def get_model(model_name, num_features, hidden_dim=128, num_classes=2):
    if model_name == 'gcn':
        return GCN(num_features, hidden_dim, num_classes)
    elif model_name == 'gat':
        return GAT(num_features, 64, num_classes)  # GAT uses less hidden dim due to multi-head
    elif model_name == 'graphsage':
        return GraphSAGE(num_features, hidden_dim, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")