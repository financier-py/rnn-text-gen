import torch
import json
from pathlib import Path
from models import TextRNN

def sample(preds, temperature=1.0):
    preds = preds / temperature
    exp_preds = torch.exp(preds)
    preds = exp_preds / torch.sum(exp_preds)
    # выбор индекса на основе вер.
    probas = torch.multinomial(preds, 1)
    return probas.item()

def generate(
        model: 'TextRNN',
        start_text='i love ',
        gen_len=500,
        temperature=0.8,
        device='cuda',
        char2int=None,
        int2char=None
) -> str:
    model.eval()
    chars = [c for c in start_text]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # скрытое состояние
    hidden = model.init_hidden(1, device)

    # начальный текст
    input_seq = torch.tensor([[char2int[c] for c in chars]]).to(device)  # type: ignore

    with torch.no_grad():
        for i in range(gen_len):
            out, hidden = model(input_seq, hidden)
            last_out = out[0, -1, :]

            char_idx = sample(last_out, temperature)
            next_char = int2char[char_idx] # type: ignore
            chars.append(next_char)

            input_seq = torch.tensor([[char_idx]]).to(device)
    
    return ''.join(chars)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = PROJECT_ROOT / 'data' / 'processed' / 'vocab.json'
WEIGHTS_PATH = PROJECT_ROOT / 'checkpoints' / 'checkpoints_best_lstm.pth'

# потестируем
if __name__ == "__main__":
    with open(VOCAB_PATH, 'r') as f:
        vocab_data = json.load(f)

    chars_list = vocab_data["chars"]
    char2int = {ch: i for i, ch in enumerate(chars_list)}
    int2char = {i: ch for i, ch in enumerate(chars_list)}
    vocab_size = vocab_data["vocab_size"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TextRNN(
        vocab_size=vocab_size,
        hidden_dim=256,  
        num_layers=2,    
        rnn_type='lstm'
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print('Веса загружены!')

    generated_text = generate(
                model, 
                gen_len=120, 
                temperature=1, 
                char2int=char2int,
                int2char=int2char
            )
    print(generated_text)