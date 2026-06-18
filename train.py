import time
import re
import pickle
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("EdinburghNLP/xsum")


MAX_ARTICLE_LEN = 300
MAX_SUMMARY_LEN = 40

### helper  functions

# cleaning
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# encoding

def encode_text(text):
  return [word2idx.get(word,UNK_IDX) for word in text.split()]

# collate_fn - helps in creating batches

def collate_fn(batch):
  articles = []
  targets = []
  decoder_inputs = []

  for article,summary in batch:
    decoder_input = summary[:-1]
    target = summary[1:]

    articles.append(
            torch.tensor(article)
        )

    decoder_inputs.append(
        torch.tensor(decoder_input)
    )

    targets.append(
        torch.tensor(target)
    )

  articles = nn.utils.rnn.pad_sequence(
      articles,
      batch_first=True,
      padding_value=PAD_IDX
  )

  targets = nn.utils.rnn.pad_sequence(
      targets,
      batch_first=True,
      padding_value=PAD_IDX
  )

  decoder_inputs = nn.utils.rnn.pad_sequence(
      decoder_inputs,
      batch_first=True,
      padding_value=PAD_IDX
  )

  return articles,decoder_inputs,targets


# training loop
def training_loop(model,train_dataloader,val_dataloader,optimizer,loss_fn,epochs):
  # training
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for article,decoder_input,target in train_dataloader:
      article = article.to(device)
      decoder_input = decoder_input.to(device)
      target = target.to(device)

      output = model(article,decoder_input)

      loss = loss_fn(
          output.reshape(-1,output.size(-1)),
          target.reshape(-1))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)


    # validation
    model.eval()

    val_loss = 0

    with torch.no_grad():

        for article, decoder_input, target in val_dataloader:

            article = article.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device)

            output = model(article, decoder_input)

            loss = loss_fn(
                output.reshape(-1, output.shape[-1]),
                target.reshape(-1)
            )

            val_loss += loss.item()

    print(
          f"Epoch {epoch+1} "
          f"Train Loss: {total_loss/len(train_dataloader):.4f} "
          f"Val Loss: {val_loss/len(val_dataloader):.4f}"
      )




PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

train_dataset_raw = dataset["train"]
val_dataset_raw = dataset["validation"]
test_dataset_raw = dataset["test"]
train_dataset_raw = train_dataset_raw.select(range(20000))
val_dataset_raw = val_dataset_raw.select(range(2000))

counter = Counter()

for sample in train_dataset_raw:
  article = clean(sample["document"])
  summary = clean(sample["summary"])

  counter.update(article.split())
  counter.update(summary.split())


max_vocab = counter.most_common(30000)

# creating 2 dict to store word2idx and idx2word
word2idx = {
    PAD: PAD_IDX,
    UNK: UNK_IDX,
    SOS: SOS_IDX,
    EOS: EOS_IDX
}

for word,_ in max_vocab :
  word2idx[word] = len(word2idx)

idx2word = {v:k for k,v in word2idx.items()}

class XSumDataset(Dataset):
  def __init__(self,data):
    self.samples = []

    for sample in data:
      article = clean(sample["document"])
      summary = clean(sample["summary"])

      article_idx = encode_text(article)[:MAX_ARTICLE_LEN]

      summary_idx = encode_text(summary)[:MAX_SUMMARY_LEN]

      summary_idx = ([SOS_IDX] + summary_idx + [EOS_IDX])

      self.samples.append((article_idx,summary_idx))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    return self.samples[idx]

# creating train datset and dataloader

train_dataset = XSumDataset(train_dataset_raw)
val_dataset = XSumDataset(val_dataset_raw)
test_dataset = XSumDataset(test_dataset_raw)

train_dataloader = DataLoader(train_dataset,batch_size=32,collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset,batch_size=32,collate_fn=collate_fn)

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

# creating a obj for seq2seq model and training it

model = Seq2Seq().to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters() ,lr = 0.001)

start = time.time()
training_loop(model,train_dataloader,val_dataloader,optimizer,loss_fn,epochs=5)

print("Time:", time.time() - start)

# saving model 
torch.save(
    model.state_dict(),
    "model.pth"
)

print("Model saved as model.pth")

with open("vocab.pkl", "wb") as f:

    pickle.dump(
        {
            "word2idx": word2idx,
            "idx2word": idx2word
        },
        f
    )

print("Vocabulary saved as vocab.pkl")

