import pickle
import re

import torch
import torch.nn as nn

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

with open("vocab.pkl", "rb") as f:

    vocab = pickle.load(f)

word2idx = vocab["word2idx"]
idx2word = vocab["idx2word"]

PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def clean(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

def encode_text(text):

    return [

        word2idx.get(
            word,
            UNK_IDX
        )

        for word in text.split()

    ]

# encoder and decoder
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            len(word2idx),
            embedding_dim=128,
            padding_idx=PAD_IDX
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            batch_first=True
        )

    def forward(self, x):

        # x
        # (batch, article_len)

        x = self.embedding(x)

        # (batch, article_len, 128)

        encoder_outputs, (hidden, cell) = self.lstm(x)

        # encoder_outputs
        # (batch, article_len, 256)

        # hidden
        # (1, batch, 256)

        return encoder_outputs, hidden, cell
class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.W1 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.W2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.V = nn.Linear(
            hidden_size,
            1
        )

    def forward(
        self,
        encoder_outputs,
        decoder_hidden
    ):

        # encoder_outputs
        # (batch, article_len, 256)

        # decoder_hidden
        # (1, batch, 256)

        decoder_hidden = decoder_hidden.permute(1, 0, 2)

        # (batch, 1, 256)

        scores = self.V(
            torch.tanh(
                self.W1(encoder_outputs)
                +
                self.W2(decoder_hidden)
            )
        )

        # (batch, article_len, 1)

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # (batch, article_len, 1)

        context = torch.sum(
            attention_weights * encoder_outputs,
            dim=1
        )

        # (batch, 256)

        return context, attention_weights


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            len(word2idx),
            embedding_dim=128,
            padding_idx=PAD_IDX
        )

        self.attention = BahdanauAttention(
            hidden_size=256
        )

        self.lstm = nn.LSTM(
            input_size=128 + 256,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(
            256,
            len(word2idx)
        )

    def forward(
        self,
        token,
        hidden,
        cell,
        encoder_outputs
    ):

        # token
        # (batch)

        embedded = self.embedding(token)

        # (batch,128)

        embedded = embedded.unsqueeze(1)

        # (batch,1,128)

        context, attn_weights = self.attention(
            encoder_outputs,
            hidden
        )

        # context
        # (batch,256)

        context = context.unsqueeze(1)

        # (batch,1,256)

        lstm_input = torch.cat(
            [embedded, context],
            dim=2
        )

        # (batch,1,384)

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        # output
        # (batch,1,256)

        prediction = self.fc(
            output.squeeze(1)
        )

        # (batch,vocab_size)

        return (
            prediction,
            hidden,
            cell,
            attn_weights
        )
# creating a model using both encoder and decoder

class Seq2Seq(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(
        self,
        article,
        decoder_input
    ):

        encoder_outputs, hidden, cell = self.encoder(
            article
        )

        outputs = []

        summary_len = decoder_input.size(1)

        for t in range(summary_len):

            token = decoder_input[:, t]

            # (batch)

            prediction, hidden, cell, _ = self.decoder(
                token,
                hidden,
                cell,
                encoder_outputs
            )

            outputs.append(prediction)

        outputs = torch.stack(
            outputs,
            dim=1
        )

        # (batch, summary_len, vocab_size)

        return outputs

model = Seq2Seq().to(device)

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

model.eval()

def summarize(text):

    model.eval()

    text = clean(text)

    article_ids = encode_text(text)

    article_ids = article_ids[:300]

    article_tensor = torch.tensor(
        article_ids
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        encoder_outputs, hidden, cell = (
            model.encoder(
                article_tensor
            )
        )

        current_token = torch.tensor(
            [SOS_IDX]
        ).to(device)

        generated = []

        for _ in range(40):

            prediction, hidden, cell, _ = (
                model.decoder(
                    current_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )

            next_token = prediction.argmax(
                dim=-1
            )

            token_id = next_token.item()

            if token_id == EOS_IDX:
                break

            generated.append(token_id)

            current_token = next_token

    words = [

        idx2word.get(
            idx,
            UNK
        )

        for idx in generated

    ]

    return " ".join(words)
