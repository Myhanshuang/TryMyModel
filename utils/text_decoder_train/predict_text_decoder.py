import torch

from main import DEVICE, D_MODEL, tokenizer, NHEAD
from model.text_decoder.my_decoder import MyTextDecoder


def generate_text_from_prompt(model, tokenizer, prompt, device, max_length=50, strategy='beam', beam_size=5,
                              top_k=50):
    """
    从给定的文本提示开始，生成后续的文本。
    """
    model.eval()

    with torch.no_grad():
        # 1. 将提示文本转换为 token ID
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # 2. 创建伪图像张量 (因为 generate 方法需要它)
        d_model = model.emb.embedding_dim
        num_patches = 50
        dummy_img = torch.zeros((1, num_patches, d_model), device=device)

        # 3. 调用模型的生成方法
        #    注意：这里的调用方式需要与您的 generate 方法的实现相匹配
        #    下面是一个自回归生成的简化示例，适配您的模型

        # 获取起始和结束 token ID
        sos_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.sep_token_id

        # --- 调用您模型中的 beam_search 或 sampling 方法 ---
        # 假设您的模型有 generate_beam_search 方法
        if strategy == 'beam':
            output_ids = model.generate(
                img=dummy_img,
                sos_token_id=sos_token_id,  # Beam search 仍从头开始
                eos_token_id=eos_token_id,
                # 注意：一个更高级的实现会把 prompt 作为 beam search 的初始状态
                max_length=max_length,
                beam_size=beam_size,
                prompt=tokenizer.encode(prompt)
            )

        # 4. 将生成的 token ID 转换回文本
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text

# --- 如何调用 ---
if __name__ == '__main__':
    # 1. 加载预训练好的模型和 tokenizer
    text_decoder = MyTextDecoder(d_model = D_MODEL, vocab_size=tokenizer.vocab_size, nhead=NHEAD)
    text_decoder.load_state_dict(torch.load("best_pretrained_decoder.pth"))
    text_decoder.to(DEVICE)

    # 2. 定义一些测试用的开头
    prompts = [
        "the two dogs are ",
        "a man in a red shirt",
        "the yellow cat",
        "who are ",
        'you are in '
    ]

    # 3. 循环生成文本并打印结果
    for p in prompts:
        # 这里我们假设模型 generate 方法已经修改为带采样的版本
        generated_sentence = generate_text_from_prompt(
            text_decoder,
            tokenizer,
            prompt=p,
            device=DEVICE,
            max_length=30
        )
        # 注意：上面的生成函数是一个简化示例，它会从<SOS>开始，而不是从 prompt 续写
        # 一个真正的续写函数会更复杂。但这个函数可以很好地测试模型的无条件生成能力。
        print(f"Generated (unconditional): {generated_sentence}")
