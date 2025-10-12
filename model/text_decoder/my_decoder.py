import torch
from torch import nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=10000, dropout=0.2):
        super().__init__()
        pos = torch.arange(start=0, end=max_len, step=1).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # print(x.shape)
        return self.dropout(x + self.pe[:, :x.size(1), :])

class MyTextDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, nhead, dropout=0.2, num_decoder=1, max_len=10000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.PE = PositionalEncoding(d_model, max_len)
        self.gating = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gate = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers= num_decoder
        )
        self.out_fc = nn.Linear(d_model, vocab_size)

    def forward(self, img, tgt, tgt_padding_mask=None):
        tgt = self.emb(tgt)
        tgt = self.PE(tgt)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        gate = self.gating(self.alpha) * self.gelu(self.gate(img))
        # gate = img
        output = self.transformer_decoder(
            tgt=tgt,
            memory=gate,
            tgt_mask=mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        logits = self.out_fc(self.dropout(output))
        return logits

    def generate(self, img, sos_token_id, eos_token_id, max_length=50, beam_size=5, prompt=None):
        """
        使用集束搜索 (Beam Search) 策略为一批图片生成描述，支持从 prompt 续写。

        Args:
            img (Tensor): 输入的图像特征张量, shape: (B, num_patches, d_model)。
            sos_token_id (int): 句子起始符的 ID。
            eos_token_id (int): 句子结束符的 ID。
            max_length (int): 生成句子的最大长度。
            beam_size (int): 集束搜索的宽度。
            prompt (list[int], optional): 已被 tokenizer编码后的提示词 ID 列表。
        """
        self.eval()

        with torch.no_grad():
            # if img.dim() == 3:
            #     img = img.unsqueeze(0),


            batch_size = img.shape[0]
            device = img.device

            # 1. 初始化 input_ids：根据有无 prompt 进行区分
            if prompt is not None:
                prompt_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
                input_ids = prompt_ids.repeat(batch_size, 1)  # 将 prompt 复制到整个批次
            else:
                input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

            # 2. 首次解码：为初始序列（prompt 或 SOS）生成第一批 beams
            logits = self.forward(img=img, tgt=input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            beam_scores, next_token_ids = torch.topk(next_token_log_probs, beam_size, dim=1, largest=True, sorted=True)

            # 3. 准备主循环的输入：将初始序列扩展 k 份，并拼接上第一批生成的词
            #    这现在正确地保留了 prompt 的上下文
            input_ids = input_ids.unsqueeze(1).repeat(1, beam_size, 1)
            input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1)

            # 4. 主循环：进行后续的解码 (这部分逻辑基本不变)
            done = (next_token_ids == eos_token_id)

            # 循环的长度需要减去已经生成的长度
            prompt_len = input_ids.shape[-1] - 1
            for _ in range(max_length - prompt_len):
                if done.all():
                    break

                current_seq_len = input_ids.shape[-1]
                current_input_ids = input_ids.view(batch_size * beam_size, current_seq_len)
                expanded_img_features = img.repeat_interleave(beam_size, dim=0)

                logits = self.forward(img=expanded_img_features, tgt=current_input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                next_token_log_probs = next_token_log_probs.view(batch_size, beam_size, -1)

                total_scores = next_token_log_probs + beam_scores.unsqueeze(-1)
                total_scores[done] = -float('Inf')

                total_scores_flat = total_scores.view(batch_size, -1)
                next_scores, next_tokens_indices = torch.topk(total_scores_flat, beam_size, dim=1, largest=True,
                                                              sorted=True)

                vocab_size = next_token_logits.shape[-1]
                beam_ids = next_tokens_indices // vocab_size
                token_ids = next_tokens_indices % vocab_size

                # 使用 gather 高效更新序列
                input_ids_flat = input_ids.view(batch_size * beam_size, -1)
                beam_indices = (beam_ids + torch.arange(batch_size, device=device).unsqueeze(1) * beam_size).view(-1)
                input_ids = input_ids_flat[beam_indices]

                input_ids = torch.cat([input_ids, token_ids.view(batch_size * beam_size, -1)], dim=-1)
                input_ids = input_ids.view(batch_size, beam_size, -1)

                beam_scores = next_scores
                done = torch.gather(done, 1, beam_ids)
                done = done | (token_ids == eos_token_id)

            # 5. 选择最终结果
            final_sequences = input_ids
            return final_sequences[:, 0, :]