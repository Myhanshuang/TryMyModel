import os.path

import torch
import transformers
from torch import nn
from transformers import AutoImageProcessor, ResNetModel

import json
from collections import Counter
from itertools import chain


class SimpleTokenizer:
    def __init__(self, captions, freq_threshold=5, max_len=50, path=None):
        self.freq_threshold = freq_threshold
        self.max_len = max_len
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.start_token_id = 1
        self.end_token_id = 2
        self.unk_token_id = 3
        if os.path.exists(path):
            self.load_tokenizer(path)
            return
        self.word2idx = {}
        self.idx2word = {}
        self.build_vocab(captions)
        self.vocab_size = len(self.idx2word)
        self.save_tokenizer('/home/acacia/PycharmProjects/PythonProject5/paper/tokenizer.json')
    def save_tokenizer(self, file_path):
        tokenizer_data = {
            "freq_threshold": self.freq_threshold,
            "max_len": self.max_len,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word
        }
        with open(file_path, "w") as f:
            json.dump(tokenizer_data, f)

    def load_tokenizer(self, file_path):
        with open(file_path, "r") as f:
            tokenizer_data = json.load(f)
        print('loading tokenizer---------------')
        self.freq_threshold = tokenizer_data["freq_threshold"]
        self.max_len = tokenizer_data["max_len"]
        self.word2idx = tokenizer_data["word2idx"]
        self.idx2word = tokenizer_data["idx2word"]
        self.vocab_size = len(self.idx2word)

    def build_vocab(self, captions):
        counter = Counter(chain.from_iterable(caption.lower().split() for caption in captions))
        words = [w for w, c in counter.items() if c >= self.freq_threshold]

        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        for i, w in enumerate(words, start=4):
            self.word2idx[w] = i
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def numericalize(self, text):
        tokens = text.lower().split()
        ids = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in tokens]
        ids = [self.word2idx[self.start_token]] + ids + [self.word2idx[self.end_token]]
        if len(ids) < self.max_len:
            ids += [self.word2idx[self.pad_token]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def decode(self, ids):
        words = [self.idx2word[i] for i in ids if i not in (0, 1, 2, 3)]
        return " ".join(words)


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        for param in self.model.parameters():
            param.requires_grad = False
        self.out_dim = 2048

    def forward(self, x):
        x = self.model(x).last_hidden_state
        # x = [batch, channel, h, w]
        x = x.permute(0, 2, 3, 1) # switch to the [batch,h, w, channel]
        x = x.view(x.size(0), -1, x.size(-1))
        return x

class SoftAttention(nn.Module):
    def __init__(self, d_model, encoder_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(encoder_dim, d_model)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.tanh = nn.ReLU()
        self.linear3 = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, a, h):
        # a = [batch, pix, channel]
        # h = [batch, hidden_state]
        attn = self.tanh(self.linear1(a) + self.linear2(h).unsqueeze(1))
        # attention = [batch, pix, d_model]
        attn_e = self.linear3(attn).squeeze(2)
        # e = [batch, pix_score]
        attn_alpha = self.softmax(attn_e)
        # alpha = [batch, p]
        attn_img = (a * attn_alpha.unsqueeze(2)).sum(1)
        return attn_img, attn_alpha

class Decoder(nn.Module):
    def __init__(self, hidden_state, encoder_dim, device, vocab_size, dropout):
        super().__init__()
        self.h_init = nn.Linear(encoder_dim, hidden_state)
        self.c_init = nn.Linear(encoder_dim, hidden_state)
        self.device = device
        self.emb = nn.Embedding(vocab_size, hidden_state)
        self.vocab_size = vocab_size
        self.attn = SoftAttention(d_model=hidden_state, encoder_dim=encoder_dim, hidden_dim=hidden_state)
        self.h_gate = nn.Linear(hidden_state, encoder_dim)
        self.lstm = nn.LSTMCell(hidden_state + encoder_dim, hidden_state)
        self.dropout = nn.Dropout(dropout)
        self.decode = nn.Linear(hidden_state, vocab_size)
        self.sigmoid = nn.Sigmoid()

    def _init_hc(self, img):
        avg_img = img.mean(dim=1)
        h = self.h_init(avg_img)
        c = self.c_init(avg_img)
        return h, c

    def forward(self, img, text):
        h, c = self._init_hc(img)

        time_steps = text.size(1) - 1
        # prev_words = torch.zeros(img.shape[0], 1).long().to(self.device)
        prev_words = text[:, 0]
        emb = self.emb(prev_words)

        preds = torch.zeros((img.shape[0], time_steps, self.vocab_size)).to(self.device)
        attn_ps = torch.zeros(img.shape[0], time_steps, img.shape[1]).to(self.device)

        for t in range(time_steps):
            attn_img, attn_p = self.attn(img, h)
            gate = self.sigmoid(self.h_gate(h))
            gate_img = attn_img * gate

            emb = emb.squeeze(1) if emb.dim() == 3 else emb
            lstm_input = torch.cat((emb, gate_img), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.decode(self.dropout(h))
            preds[:, t] = output
            attn_ps[:, t] = attn_p

            if not self.training:
                # greedy search! no beam search
                emb = self.emb(output.max(1)[1].reshape(img.shape[0], 1))
            else :
                emb = self.emb(text[:, t + 1])
        return preds, attn_ps

class PaperModel(nn.Module):
    def __init__(self, hidden_state, encoder_dim, device, vocab_size, dropout):
        super().__init__()
        self.encoder = VisionEncoder()
        self.decoder = Decoder(hidden_state=hidden_state, encoder_dim=encoder_dim, device=device, vocab_size=vocab_size, dropout=dropout)
        self.name = 'paper'
    def forward(self, img, text):
        img_feature = self.encoder(img)

        preds, attn_ps = self.decoder(img_feature, text)

        return preds, attn_ps

    def generate(self, pixel_values, max_length, num_beams):
        """
        使用 Beam Search 生成图像描述。
        这个实现是向量化的，可以正确处理 batch。
        """
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        # 1. 将图像编码一次，并为每个 beam 复制
        img_features = self.encoder(pixel_values)  # Shape: [batch_size, num_pixels, encoder_dim]
        img_features = img_features.expand(batch_size, num_beams, -1, -1).reshape(batch_size * num_beams,
                                                                                  img_features.size(1),
                                                                                  img_features.size(2))

        # 2. 初始化 LSTM 的隐藏状态 (h, c)，并为每个 beam 复制
        h, c = self.decoder._init_hc(img_features)  # Shape: [batch_size * num_beams, hidden_dim]

        # 3. 初始化序列和分数
        # beam_sequences 存储了每个 beam 正在生成的序列
        beam_sequences = torch.full((batch_size, num_beams, max_length), self.decoder.emb.padding_idx, dtype=torch.long,
                                    device=device)
        # 使用 <start> token 初始化所有序列的第一个词
        beam_sequences[:, :, 0] = self.decoder.emb.weight.requires_grad  # tokenizer.start_token_id

        # beam_scores 存储了每个 beam 的累计对数概率分数
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # 确保只有第一个 beam 在开始时是活跃的

        # 当前时间步的输入词 (decoder input)
        # 在第一个时间步，所有 beam 的输入都是 <start> token
        prev_words = torch.full((batch_size * num_beams, 1), self.decoder.emb.weight.requires_grad, dtype=torch.long,
                                device=device)  # tokenizer.start_token_id

        # 跟踪已完成的序列
        completed_sequences = []

        for t in range(1, max_length):
            # emb.shape: [batch_size * num_beams, embedding_dim]
            emb = self.decoder.emb(prev_words).squeeze(1)

            # --- 以下是 Decoder 的单步前向传播 ---
            attn_img, _ = self.decoder.attn(img_features, h)
            gate = self.decoder.sigmoid(self.decoder.h_gate(h))
            gate_img = attn_img * gate
            lstm_input = torch.cat((emb, gate_img), dim=1)
            h, c = self.decoder.lstm(lstm_input, (h, c))
            logits = self.decoder.decode(h)  # Shape: [batch_size * num_beams, vocab_size]

            # --- Beam Search 核心逻辑 ---
            log_probs = torch.log_softmax(logits, dim=-1)  # 对数概率

            # 将 log_probs 与之前的 beam 分数相加，得到所有候选序列的总分数
            # log_probs.shape: [batch_size * num_beams, vocab_size]
            # beam_scores.view(-1, 1): [batch_size * num_beams, 1]
            candidate_scores = beam_scores.view(-1, 1) + log_probs

            # 展平分数，以便在所有 beam 的所有可能 next_token 中选择 top-k
            # flat_candidate_scores.shape: [batch_size, num_beams * vocab_size]
            flat_candidate_scores = candidate_scores.view(batch_size, -1)

            # 在每个样本内部，独立地选出分数最高的 num_beams 个候选
            top_scores, top_indices = torch.topk(flat_candidate_scores, num_beams, dim=-1)

            # 更新 beam 分数
            beam_scores = top_scores

            # 从 top_indices 解码出是哪个 beam 产生了哪个词
            beam_indices = top_indices // self.decoder.vocab_size  # 确定来源 beam (0, 1, ..., num_beams-1)
            token_indices = top_indices % self.decoder.vocab_size  # 确定生成的词的 ID

            # 根据 beam_indices 重新整理 beam_sequences, h, c
            # 这一步是关键，确保每个新 beam 都继承了正确的历史序列和隐藏状态
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_beams)
            beam_sequences = beam_sequences[batch_indices, beam_indices]
            h = h.view(batch_size, num_beams, -1)[batch_indices, beam_indices].reshape(batch_size * num_beams, -1)
            c = c.view(batch_size, num_beams, -1)[batch_indices, beam_indices].reshape(batch_size * num_beams, -1)

            # 将新生成的词添加到序列中
            beam_sequences[:, :, t] = token_indices

            # 更新下一个时间步的输入
            prev_words = token_indices.view(batch_size * num_beams, 1)

            # 检查是否有 beam 生成了 <end> token，并提前终止
            is_end = (token_indices == 2).any()  # tokenizer.end_token_id
            if is_end:
                break

        # 返回分数最高的那个序列 (在 batch 中每个样本都选第 0 个 beam)
        best_sequences = beam_sequences[:, 0, :]
        return best_sequences