from torch import nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, img_encoder, text_decoder, name):
        super().__init__()
        self.encoder = img_encoder
        self.decoder = text_decoder
        self.name = name
    def forward(self, img, tgt, caption_mask=None):
        img = self.encoder(img)
        img = img.unsqueeze(1)
        logits = self.decoder(img, tgt, caption_mask)
        return logits

    def greedy_generate(self, img, sos_token_id, eos_token_id, max_length=100):
        self.eval()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        batch_size = img.shape[0]
        img = self.encoder(img).unsqueeze(1)

        decoder_input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long).to(img.device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                logits = self.decoder(img, decoder_input_ids)

                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=1)

                if (next_token_id == eos_token_id).all():
                    break

        return decoder_input_ids

    def beam_generate(self, img, sos_token_id, eos_token_id, max_length=100, beam_size=5):
        """
        使用集束搜索 (Beam Search) 策略为一批图片生成描述。
        此实现支持 batch_size > 1。
        """
        self.eval()

        with torch.no_grad():
            img = img.unsqueeze(0)
            batch_size = img.shape[0]
            device = img.device

            # 1. 编码图像（只需一次）
            img_features = self.encoder(img).unsqueeze(1)  # Shape: (B, 1, num_patches, embed_dim)

            # --- 步骤 A: 处理第一次解码 (特殊情况) ---

            # a. 初始输入只有 SOS token
            #    input_ids shape: (B, 1)
            input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

            # b. 第一次前向传播
            #    img_features shape: (B, ...), input_ids shape: (B, 1)
            logits = self.decoder(img_features, input_ids)
            next_token_logits = logits[:, -1, :]  # Shape: (B, vocab_size)
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            # c. 从每个 batch item 中选出 top-k 个起始词，构成最初的 beams
            #    beam_scores, next_token_ids shape: (B, beam_size)
            beam_scores, next_token_ids = torch.topk(next_token_log_probs, beam_size, dim=1, largest=True, sorted=True)

            # d. 准备下一次循环的输入
            #    input_ids shape: (B, beam_size, 2)
            input_ids = torch.cat([
                torch.full((batch_size, beam_size, 1), sos_token_id, dtype=torch.long, device=device),
                next_token_ids.unsqueeze(-1)
            ], dim=-1)

            # --- 步骤 B: 开始主循环进行后续解码 ---

            done = (next_token_ids == eos_token_id)  # 追踪已完成的 beams

            # 循环从第二步开始
            for _ in range(max_length - 2):
                if done.all():
                    break

                # a. 准备模型输入
                #    current_input_ids shape: (B * k, current_seq_len)
                current_input_ids = input_ids.view(batch_size * beam_size, -1)

                # b. 扩展图像特征以匹配 beam 数量
                expanded_img_features = img_features.repeat_interleave(beam_size, dim=0)

                # c. 模型前向传播
                logits = self.decoder(expanded_img_features, current_input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                next_token_log_probs = next_token_log_probs.view(batch_size, beam_size, -1)

                # d. 计算总分
                total_scores = next_token_log_probs + beam_scores.unsqueeze(-1)

                # e. 将已完成的 beam 的分数设为极低，防止其被再次扩展
                total_scores[done] = -1e9

                # f. 展平并选择 top-k
                total_scores_flat = total_scores.view(batch_size, -1)
                next_scores, next_tokens_indices = torch.topk(total_scores_flat, beam_size, dim=1, largest=True,
                                                              sorted=True)

                # g. 解码 beam_id 和 token_id
                vocab_size = next_token_logits.shape[-1]
                beam_ids = next_tokens_indices // vocab_size
                token_ids = next_tokens_indices % vocab_size

                # h. 使用 gather 高效更新序列
                #    input_ids shape: (B, k, seq_len)
                input_ids = input_ids.view(batch_size, beam_size, -1)
                input_ids = torch.gather(input_ids, 1, beam_ids.unsqueeze(-1).expand_as(input_ids))

                # 拼接新词
                input_ids = torch.cat([input_ids, token_ids.unsqueeze(-1)], dim=-1)

                # i. 更新分数和完成状态
                beam_scores = next_scores
                done = torch.gather(done, 1, beam_ids)
                done = done | (token_ids == eos_token_id)

            # 5. 选择最终结果
            #    将 input_ids 转换回 (B, k, seq_len)
            final_sequences = input_ids.view(batch_size, beam_size, -1)
            # 返回每个 batch item 分数最高的那个序列
            return final_sequences[:, 0, :]


class ShowAttendTellModel(nn.Module):
    """
    组合了 AttentionEncoder 和 AttentionDecoder，构成了完整的 "Show, Attend and Tell" 模型。
    """

    def __init__(self, d_model, embed_dim, decoder_dim, vocab_size, name="ShowAttendTell"):
        super(ShowAttendTellModel, self).__init__()
        encoder = AttentionDecoder(d_model, embed_dim, decoder_dim, vocab_size)
        decoder = AttentionEncoder()
        self.encoder = encoder
        self.decoder = decoder
        self.name = name

    def forward(self, images, encoded_captions, caption_lengths):
        """
        训练时的前向传播。

        Args:
            images (Tensor): 输入的图像。
            encoded_captions (Tensor): 经过编码的、带padding的标题。
            caption_lengths (list): 每个标题的真实长度。

        Returns:
            predictions (Tensor): 模型的输出 logits。
            alphas (Tensor): 每一步生成的注意力权重。
        """
        encoder_out = self.encoder(images)
        predictions, alphas = self.decoder(encoder_out, encoded_captions, caption_lengths)
        return predictions, alphas

    def generate(self, images, tokenizer, max_len=50):
        """
        用于推理的生成函数。
        注意：这个经典模型的推理逻辑与Hugging Face的 .generate() 不同，
        需要一个自定义的、逐词生成的循环。
        """
        # 完整的推理逻辑（beam search）比较复杂，这里提供一个贪心搜索的简化实现
        # 以便快速验证模型效果。

        self.eval()  # 切换到评估模式

        encoder_out = self.encoder(images)
        batch_size = encoder_out.size(0)

        # 获取起始和结束token的ID
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        end_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

        # 初始化生成的序列
        generated_sequences = torch.LongTensor([[start_token_id]] * batch_size).to(images.device)

        with torch.no_grad():
            h, c = self.decoder.init_hidden_state(encoder_out)

            for t in range(max_len):
                # 获取上一个时间步的词嵌入
                embeddings = self.decoder.embedding(generated_sequences[:, t])

                # 计算注意力
                awe, _ = self.decoder.attention(encoder_out, h)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                awe = gate * awe

                # 运行LSTM单元
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

                # 预测下一个词
                preds = self.decoder.fc(h)
                next_word_inds = preds.argmax(dim=1)

                # 拼接到序列中
                generated_sequences = torch.cat([generated_sequences, next_word_inds.unsqueeze(1)], dim=1)

                # 如果所有批次都生成了结束符，则提前停止
                if (next_word_inds == end_token_id).all():
                    break

        return generated_sequences
