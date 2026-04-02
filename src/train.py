import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from pathlib import Path
from tqdm import tqdm

from dataset import get_dataloader
from models import TextRNN

CONFIG = {
    'rnn_type': 'lstm',       # Выбор архитектуры: 'rnn', 'gru', 'lstm'
    'seq_len': 128,           # Длина окна текста
    'batch_size': 512,        # Количество окон в батче
    'embed_dim': 64,          # Размерность букв
    'hidden_dim': 256,        # Размер памяти сети (aka скрытого слоя)
    'num_layers': 2,          # Количество слоев RNN
    'drop_prob': 0.2,         # Регуляризация
    'lr': 0.006,              # Learning rate
    'epochs': 20,             # Количество проходов по всему тексту
    'clip': 5.0               # Макс. знач. градиента
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'checkpoints'
MODELS_DIR.mkdir(parents=True, exist_ok=True) # у меня падал проект, поэтому добавил

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # у меня только cpu увы

def train_model():
    # для красивого дашборда, по совету нейронки
    wandb.init(
        project='tolstoy-text-gen',
        name=f"run-{CONFIG['rnn_type']}-h{CONFIG['hidden_dim']}",
        config=CONFIG
    )

    # подготавливаем данные
    dataloader, vocab_size = get_dataloader(
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size']
    )

    wandb.config.update({'vocab_size': vocab_size})

    # Инициализируем модель, ф-ию потерь и оптимизатор
    model = TextRNN(
        vocab_size=vocab_size,
        embed_dim=CONFIG['embed_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        rnn_type=CONFIG['rnn_type'],
        drop_prob=CONFIG['drop_prob']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    best_loss = float('inf')

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()

        # в начале каждой эпохи hidden = 0
        hidden = model.init_hidden(CONFIG['batch_size'], device)

        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{CONFIG['epochs']}", leave=False)

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            if CONFIG['rnn_type'] == 'lstm':
                hidden = tuple([h.detach() for h in hidden])
            else:
                hidden = hidden.detach() # type: ignore
            
            optimizer.zero_grad()

            output, hidden = model(x, hidden)

            loss = criterion(output.view(-1, vocab_size), y.view(-1))

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['clip'])

            optimizer.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch": epoch, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = MODELS_DIR / f"best_{CONFIG['rnn_type']}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(str(checkpoint_path))
            print(f"Сохраняем новую лучшую модельку: {checkpoint_path.name}")
        
    print('Обучение закончено :)')
    wandb.finish()

if __name__ == '__main__':
    train_model()