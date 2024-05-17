import torch
import torch.nn as nn

# from .build import HEADS
from .base import BaseHead
from builder.builder import HeadRegistry

def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.normal_(module.bias, bias)

@HeadRegistry.register_module()
class EmptyHead(BaseHead):

    def __init__(self,num_classes=None,in_channels=None):
        super().__init__(num_classes, in_channels)
        pass

    def init_weights(self):
        pass

    def forward(self, x):
        return x


@HeadRegistry.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

    def forward_dummy(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score


# @HEADS.register_module()
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)


# @HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HeadRegistry.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


# @HEADS.register_module()
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)


# @HEADS.register_module()
class GCNHead_GRU(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_state_dim,
                 part_dim=2,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)
        # self.hidden_state_dim = in_channels
        self.hidden_state_dim = hidden_state_dim
        self.gru = nn.GRU(in_channels, self.hidden_state_dim)
        self.P_dim = part_dim
        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.hidden_state_dim, num_classes)
        pass

    def forward(self, x, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            x = x.reshape(int(x.size(0) / self.P_dim), self.P_dim, x.size(1))
            x = x.mean(dim=1)
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        # return cls_score, output[0]
        # output[0] 用于tsne的可视化
        return cls_score

    def confirm_forward(self, x, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            x = x.reshape(int(x.size(0) / self.P_dim), self.P_dim, x.size(1))
            x = x.mean(dim=1)
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            # self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score

    def forward_dummy(self, x, restart_batch=True):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim))
            self.hidden = h_n
        else:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score


# @HEADS.register_module()
class GCNHead_Part(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 part_dim=2,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)
        self.P_dim = part_dim

    def forward(self, x, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)

        x = x.reshape(int(x.size(0) / self.P_dim), self.P_dim, x.size(1))
        x = x.mean(dim=1)

        cls_score = self.fc_cls(x)
        return cls_score


# @HEADS.register_module()
class STN_GCNHead_GRU(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)
        self.hidden_state_dim = in_channels
        self.gru = nn.GRU(in_channels, self.hidden_state_dim)

    def forward(self, x, s_action=None, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            part_data = []
            pool = nn.AdaptiveAvgPool2d(1)
            for p_d in x:
                N, M, C, T, V = p_d.shape
                p_d = p_d.reshape(N * M, C, T, V)
                p_d = pool(p_d)
                p_d = p_d.reshape(N, M, C)
                p_d = p_d.mean(dim=1)
                part_data.append(p_d)
            part_data = torch.stack(part_data, 0)
            part_data = part_data.permute(1, 0, 2).contiguous()
            x = torch.einsum('bp,bpc->bc', s_action, part_data)
        else:
            pool = nn.AdaptiveAvgPool2d(1)
            N, M, C, T, V = x.shape
            x = x.reshape(N * M, C, T, V)

            x = pool(x)
            x = x.reshape(N, M, C)
            x = x.mean(dim=1)

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score

    def confirm_forward(self, x, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            # self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score

    def val_forward(self, x, restart_batch=False):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
            self.hidden = h_n
        else:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score

    def forward_dummy(self, x, restart_batch=True):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c

        if self.dropout is not None:
            x = self.dropout(x)
        if restart_batch:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)),
                                   torch.zeros(1, x.size(0), self.hidden_state_dim))
            self.hidden = h_n
        else:
            output, h_n = self.gru(x.view(1, x.size(0), x.size(1)), self.hidden)
            self.hidden = h_n
        cls_score = self.fc_cls(output[0])
        return cls_score
