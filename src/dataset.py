import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

PROJECT_PATH = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PROJECT_PATH / 'data' / 'processed' / 'clean_text.txt'
VOCAB_PATH = PROJECT_PATH / 'data' / 'processed' / 'vocab.json'

class TextDataset(Dataset):
    def __init__(self, text_path: Path, vocab_path: Path, seq_len: int = 128):
        self.seq_len = seq_len
        self.char2int, self.int2char, self.vocab_size = self._load_vocab(vocab_path)
        self.encoded_text = self._encode_text(text_path)
    
    def _load_vocab(self, path: Path) -> tuple[dict[str, int], dict[int, str], int]:
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        chars = vocab_data['chars']
        char2int = {ch: i for i, ch in enumerate(chars)}
        int2char = {i: ch for i, ch in enumerate(chars)}

        return char2int, int2char, vocab_data['vocab_size']
    
    def _encode_text(self, text_path: Path) -> list[int]:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [self.char2int[c] for c in text]

    def __len__(self) -> int:
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        chunk = self.encoded_text[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(
        text_path: Path = CLEAN_DATA_PATH,
        vocab_path: Path = VOCAB_PATH,
        seq_len: int = 128,
        batch_size: int = 64, 
        shuffle: bool = True
) -> tuple[DataLoader, int]:

    dataset = TextDataset(text_path, vocab_path, seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2
    )

    return dataloader, dataset.vocab_size

# Просто тесты и проверка
if __name__ == '__main__':
    dataloader, vocab_size = get_dataloader(seq_len=50, batch_size=2)

    x_batch, y_batch = next(iter(dataloader))

    print(f"X: {x_batch.shape} -> [batch_size, seq_length]")
    print(f"Y: {y_batch.shape} -> [batch_size, seq_length]\n")
    
    dataset = dataloader.dataset
    decoded_x = "".join([dataset.int2char[i.item()] for i in x_batch[0]]) # pyright: ignore[reportAttributeAccessIssue]
    decoded_y = "".join([dataset.int2char[i.item()] for i in y_batch[0]]) # pyright: ignore[reportAttributeAccessIssue]
    
    print('Расшифруем, дабы посмотреть')
    print('X:', decoded_x)
    print('Y:', decoded_y)

    # норм, работает !