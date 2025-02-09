import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from transformers import AutoModelForCausalLM
except ImportError:
    raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
try:
    dtype = torch.bfloat16
    torch.zeros(1, dtype=dtype, device=device)
except RuntimeError:
    dtype = torch.float16


class JanusProModelLoaderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
            },
        }

    RETURN_TYPES = ("vl_gpt", "vl_chat_processor")
    RETURN_NAMES = ("vl_gpt", "vl_chat_processor")
    FUNCTION = "node_function"
    CATEGORY = "Fair/deepseek"

    def node_function(self, model_name):

        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_dir = os.path.join(comfy_path, "models", "deepseek-ai", os.path.basename(model_name))

        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)

        vl_gpt = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        vl_gpt = vl_gpt.to(dtype).to(device).eval()

        return (vl_gpt, vl_chat_processor)


class JanusProMultimodalUnderstandingNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vl_gpt": ("vl_gpt",),
                "vl_chat_processor": ("vl_chat_processor",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "seed": ("INT", {"default": 666666666666666, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understanding_image"
    CATEGORY = "Fair/deepseek"

    def understanding(self, vl_gpt, vl_chat_processor, image, question, temperature, top_p, max_new_tokens):
        image = (torch.clamp(image, 0, 1) * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(image, mode="RGB")

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [pil_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = vl_chat_processor(conversations=conversation, images=[pil_image], force_batchify=True).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        tokenizer = vl_chat_processor.tokenizer
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        return answer

    def understanding_image(self, vl_gpt, vl_chat_processor, image, question, temperature, top_p, max_new_tokens, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        answers = []
        # [Batch, Channel, Height, Width]
        if len(image.shape) == 4:
            tensors = torch.unbind(image, dim=0)
            for i in tensors:
                image = i

                answer = self.understanding(vl_gpt, vl_chat_processor, image, question, temperature, top_p, max_new_tokens)
                answers.append(answer)

        # [Channel, Height, Width]
        else:
            answer = self.understanding(vl_gpt, vl_chat_processor, image, question, temperature, top_p, max_new_tokens)
            answers.append(answer)

        return (answers,)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


class JanusProImageGenerationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vl_gpt": ("vl_gpt",),
                "vl_chat_processor": ("vl_chat_processor",),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful photo of"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "parallel_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "cfg_weight": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "image_token_num_per_image": ("INT", {"default": 576, "min": 1, "max": 576}),
                "img_size": ("INT", {"default": 384, "min": 1, "max": 384}),
                "patch_size": ("INT", {"default": 16, "min": 1, "max": 16}),
                "seed": ("INT", {"default": 666666666666666, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "node_function"
    CATEGORY = "Fair/deepseek"

    def generate(
        self,
        vl_gpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        temperature: float = 1,
        parallel_size: int = 16,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

        for i in range(image_token_num_per_image):
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2, 0, 1)
        images = torch.from_numpy(dec).float()

        return (images,)

    def node_function(
        self,
        vl_gpt,
        vl_chat_processor,
        prompt,
        temperature,
        parallel_size,
        cfg_weight,
        image_token_num_per_image,
        img_size,
        patch_size,
        seed=666666666666666,
    ):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        return self.generate(
            vl_gpt,
            vl_chat_processor,
            prompt,
            temperature,
            parallel_size,
            cfg_weight,
            image_token_num_per_image,
            img_size,
            patch_size,
        )

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed
