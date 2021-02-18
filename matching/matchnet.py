import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.conv_block import ConvBlock
from utils.common import pairwise_distances, split_support_query_set

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classifier(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.block1 = ConvBlock(in_channel, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block2 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block3 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block4 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)

        self.init_params()

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(x.size(0), -1)
        return out

    def init_params(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)


# from https://github.com/oscarknagg/few-shot
class BidirectionalLSTM(nn.Module):
    def __init__(self, size, layers):
        """
        This used to generate FCE of the support set
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=size, num_layers=layers, hidden_size=size, bidirectional=True)

    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, _ = self.lstm(inputs, None)  # output, (hn, cn)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        output = forward_output + backward_output + inputs
        return output


# from https://github.com/oscarknagg/few-shot
class AttentionLSTM(nn.Module):
    def __init__(self, size, unrolling_steps):
        """
        This used to generate FCE of the query set
        """
        super().__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size, hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise (ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).to(device)
        c = torch.zeros(batch_size, embedding_dim).to(device)

        for k in range(self.unrolling_steps):
            # cf. equation (4) of appendix A.2
            h = h_hat + queries

            # cf. equation (6) of appendix A.2
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)

            # cf. equation (5)
            readout = torch.mm(attentions, support)

            # cf. equation (3)
            h_hat, c = self.lstm_cell(queries, (h + readout, c))

        h = h_hat + queries

        return h


class MatchingNetworks(nn.Module):
    def __init__(self, n_way, k_support, k_query, k_query_val, in_channel, lstm_layers, lstm_input_size,
                 unrolling_steps,
                 fce=True,
                 distance_fn='cosine'):
        """
        n_way = Number of classes in support set.
        k_support = Number of examples per class in support set. k_shot
        k_query = Number of examples per class in query set. k_shot
        in_channel: Color channel. Omniglot = 1, miniImageNet = 3
        lstm_layers: Number of lstm layers in the bidirectional LSTM.
        lstm_input_size: Input size of two LSTM. Omniglot = 64, miniImageNet = 1600
        unrolling_steps: Number of unrolling step.
        """
        super().__init__()
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.k_query_val = k_query_val
        self.fce = fce
        self.distance_fn = distance_fn
        self.is_train = True

        self.classifier = Classifier(in_channel)
        if self.fce:
            self.g = BidirectionalLSTM(lstm_input_size, lstm_layers).to(device)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps).to(device)

    def forward(self, x, y):
        embedding = self.classifier(x)

        num_query = self.k_query if self.is_train else self.k_query_val

        support, query, y_query = split_support_query_set(embedding, y, self.n_way, self.k_support, num_query)

        if self.fce:
            support = self.g(support.unsqueeze(1)).squeeze(1)
            query = self.f(support, query)

        distance = pairwise_distances(query, support, self.distance_fn)

        attention = F.softmax(-distance, dim=1)

        _scatter = torch.arange(0, self.n_way, 1 / self.k_query).long().to(device).unsqueeze(-1)
        y_one_hot = torch.zeros(self.k_query * self.n_way, self.n_way).to(device).scatter(1, _scatter, 1)
        y_pred = torch.mm(attention, y_one_hot)

        return y_pred.clamp(1e-8, 1 - 1e-8), y_query

    def custom_train(self):
        self.is_train = True

    def custom_eval(self):
        self.is_train = False


if __name__ == '__main__':
    import torch

    x = torch.rand((30, 3, 84, 84))
    y1 = torch.LongTensor([x for x in range(3) for _ in range(5)])
    y2 = torch.LongTensor([x for x in range(3) for _ in range(5)])
    y = torch.cat((y1, y2))

    model = MatchingNetworks(3, 5, 5, 5, 3, 1, 1600, 2)

    y_pred, y = model(x, y)

    loss = torch.nn.CrossEntropyLoss()(y_pred, y)

    acc = y_pred.argmax(dim=1).eq(y).float().mean()
