import torch.nn as nn
import torch

from utils.conv_block import ConvBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classifier(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block1 = ConvBlock(input_channel, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block2 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block3 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)
        self.block4 = ConvBlock(64, 64, 3, activation='ReLU', max_pool=2, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(x.size(0), -1)
        return out


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
        output, (hn, cn) = self.lstm(inputs, None)

        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        output = forward_output + backward_output + inputs
        return output, hn, cn


# from https://github.com/oscarknagg/few-shot
class AttentionLSTM(nn.Module):
    def __init__(self, size, unrolling_step):
        """
        This used to generate FCE of the query set
        """
        super().__init__()
        self.unrolling_step = unrolling_step
        self.lstm_cell = nn.LSTMCell(input_size=size, hidden_size=size)

    def forward(self, support, queries):
        # Get embedding dimension, d
        if support.shape[-1] != queries.shape[-1]:
            raise (ValueError("Support and query set have different embedding dimension!"))

        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]

        h_hat = torch.zeros_like(queries).to(device).double()
        c = torch.zeros(batch_size, embedding_dim).to(device).double()

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
    def __init__(self, input_channel, lstm_layers, lstm_input_size, unrolling_step, fce=True):
        """
        input_channel: color channel. Omniglot = 1, miniImageNet = 3
        lstm_layers: Number of lstm layers in the bidirectional LSTM
        lstm_input_size: Input size of two LSTM. Omniglot = 64, miniImageNet = 1600
        """
        super().__init__()
        self.fce = fce
        self.classifier = Classifier(input_channel)

        if self.fce:
            self.g = BidirectionalLSTM(lstm_input_size, lstm_layers).to(device)
            self.f = AttentionLSTM(lstm_input_size, unrolling_step).to(device)

    def forward(self):
        pass


if __name__ == '__main__':
    import torch

    img = torch.rand((1, 3, 84, 84))
    model = Classifier(3)
    y_pred = model(img)
    print(y_pred.shape)
