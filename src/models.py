import torch
import torch.nn as nn

class TextRNN(nn.Module):
    """
    Рекуррентная сеть для генерации текста.
    Поддерживает три архитектуры: 'rnn', 'gru', 'lstm'
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 256,
            num_layers = 1,
            drop_prob: float = 0.2,
            rnn_type: str = 'lstm'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_classes = {
            'rnn': nn.RNN,
            'gru': nn.GRU,
            'lstm': nn.LSTM
        }

        if self.rnn_type not in rnn_classes:
            raise ValueError(f'{rnn_type} не поддерживается. Доступны только {rnn_classes.keys()}')
        
        RNNClass = rnn_classes[self.rnn_type]
        self.rnn = RNNClass(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=drop_prob if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def _init_weights(self) -> None:
        # Проинициализируем Embedding и Linear (Xavier Uniform)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        if self.rnn_type == 'lstm':
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    # вторая четверть bias в LSTM это forget gate
                    start, end = n // 4, n // 2
                    # инициализирую единицами, дабы сохранить всю "память"
                    param.data[start:end].fill_(1.0)
                elif 'weight' in name:
                    # собственные значения = 1, чтобы сигнал не затух или не взорвался
                    nn.init.orthogonal_(param)
    
    def forward(
            self,
            x: torch.Tensor,
            hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

    def init_hidden(
            self, batch_size: int, device: torch.device
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return h0, c0
        return h0