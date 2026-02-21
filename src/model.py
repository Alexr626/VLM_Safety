"""
VLM wrapper for activation and attention extraction experiments.

Primary target: llava-hf/llava-1.5-7b-hf
  - LlavaForConditionalGeneration (transformers >= 4.37)
  - CLIP ViT-L/14@336px → 576 image tokens (24×24 grid)
  - LLaMA-2-based LLM backbone (32 layers, hidden_dim=4096)
  - Image token ID: 32000

Secondary (optional): llava-hf/llava-v1.6-vicuna-7b-hf
  - LlavaNextForConditionalGeneration

Usage
-----
    wrapper = VLMWrapper("llava-hf/llava-1.5-7b-hf").load()

    # Multimodal forward pass
    hidden, attentions = wrapper.forward_vl(image, text, output_attentions=True)

    # Text-only forward pass
    hidden, _ = wrapper.forward_text(text_prompt)

    # Caption generation
    caption = wrapper.generate_caption(image)
"""

import gc
import torch
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image


# ── Model registry ────────────────────────────────────────────────────────────

_MODEL_CONFIGS = {
    "llava-hf/llava-1.5-7b-hf": {
        "model_class": "LlavaForConditionalGeneration",
        "num_image_tokens": 576,   # CLIP ViT-L/14@336 → 24×24 patches
        "prompt_template": "USER: <image>\n{text}\nASSISTANT:",
        "text_only_template": "USER: {text}\nASSISTANT:",
        "caption_prompt": "Describe this image in detail.",
    },
    "llava-hf/llava-v1.6-vicuna-7b-hf": {
        "model_class": "LlavaNextForConditionalGeneration",
        "num_image_tokens": None,  # dynamic; detected at runtime
        "prompt_template": "USER: <image>\n{text}\nASSISTANT:",
        "text_only_template": "USER: {text}\nASSISTANT:",
        "caption_prompt": "Describe this image in detail.",
    },
}

_DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


class VLMWrapper:
    """
    Unified wrapper providing:
      - Model + processor loading
      - Multimodal forward passes (image + text)
      - Text-only forward passes
      - Caption generation
      - Image token position detection
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        torch_dtype=torch.float16,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        cfg = _MODEL_CONFIGS.get(model_id, _MODEL_CONFIGS[_DEFAULT_MODEL])
        self._model_class_name = cfg["model_class"]
        self._num_image_tokens_cfg = cfg["num_image_tokens"]
        self._prompt_template = cfg["prompt_template"]
        self._text_only_template = cfg["text_only_template"]
        self._caption_prompt = cfg["caption_prompt"]

        self.model = None
        self.processor = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "VLMWrapper":
        """Load model and processor. Returns self for chaining."""
        from transformers import AutoProcessor

        print(f"Loading processor from '{self.model_id}'...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        print(f"Loading model ({self._model_class_name}) from '{self.model_id}'...")
        ModelClass = self._get_model_class()
        self.model = ModelClass.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()

        print(f"  num_hidden_layers : {self.num_layers}")
        print(f"  hidden_size       : {self.hidden_dim}")
        print(f"  num_image_tokens  : {self.num_image_tokens}")
        print(f"  image_token_id    : {self.image_token_id}")
        return self

    def _get_model_class(self):
        from transformers import (
            LlavaForConditionalGeneration,
            LlavaNextForConditionalGeneration,
        )
        mapping = {
            "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
            "LlavaNextForConditionalGeneration": LlavaNextForConditionalGeneration,
        }
        return mapping[self._model_class_name]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the LLM backbone."""
        return self.model.config.text_config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.model.config.text_config.num_attention_heads

    @property
    def image_token_id(self) -> int:
        """Token ID used as the <image> placeholder in input_ids."""
        return self.model.config.image_token_index

    @property
    def num_image_tokens(self) -> int:
        """Number of visual tokens injected per image after projection."""
        if self._num_image_tokens_cfg is not None:
            return self._num_image_tokens_cfg
        # Detect at runtime from config
        vision_cfg = self.model.config.vision_config
        img_size = getattr(vision_cfg, "image_size", 336)
        patch_size = getattr(vision_cfg, "patch_size", 14)
        n = (img_size // patch_size) ** 2
        return n

    @property
    def model_name(self) -> str:
        """Short name for use in file paths."""
        return self.model_id.split("/")[-1]

    @property
    def device(self):
        return next(self.model.parameters()).device

    # ── Input preparation ─────────────────────────────────────────────────────

    def _prepare_vl(self, text: str, image: Image.Image) -> dict:
        """Prepare multimodal inputs (image + text)."""
        prompt = self._prompt_template.format(text=text)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _prepare_text(self, text: str) -> dict:
        """Prepare text-only inputs (no image)."""
        prompt = self._text_only_template.format(text=text)
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    # ── Image token position detection ────────────────────────────────────────

    def get_image_token_span(self, input_ids: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        """
        Return (img_start, img_end) in the *expanded* sequence.

        In the expanded sequence the single <image> placeholder is replaced by
        `num_image_tokens` consecutive visual feature tokens.

        Returns (None, None) if no image token is found.
        """
        img_id = self.image_token_id
        matches = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            return None, None
        img_pos = int(matches[0].item())
        return img_pos, img_pos + self.num_image_tokens

    def get_text_token_positions(
        self,
        input_ids: torch.Tensor,
    ) -> List[int]:
        """
        Return list of positions in the expanded sequence that are NOT image tokens.
        These represent the text query positions used for cross-modal attention.
        """
        img_start, img_end = self.get_image_token_span(input_ids)
        orig_len = input_ids.shape[1]
        # Expanded length: replace 1 image placeholder with num_image_tokens tokens
        if img_start is None:
            expanded_len = orig_len
        else:
            expanded_len = orig_len - 1 + self.num_image_tokens

        if img_start is None:
            return list(range(expanded_len))

        return (
            list(range(img_start)) +                        # before image
            list(range(img_end, expanded_len))              # after image
        )

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward_vl(
        self,
        image: Image.Image,
        text: str,
        output_attentions: bool = False,
    ) -> Tuple[tuple, Optional[tuple]]:
        """
        Multimodal forward pass.

        Returns:
            hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden_dim)
            attentions:    tuple of (num_layers) tensors, each (1, num_heads, seq_len, seq_len)
                           or None if output_attentions=False.
        """
        inputs = self._prepare_vl(text, image)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        hidden = outputs.hidden_states
        attns = outputs.attentions if output_attentions else None
        return hidden, attns, inputs.get("input_ids")

    def forward_text(
        self,
        text: str,
        output_attentions: bool = False,
    ) -> Tuple[tuple, Optional[tuple]]:
        """
        Text-only forward pass (no image).

        Returns same structure as forward_vl.
        """
        inputs = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        hidden = outputs.hidden_states
        attns = outputs.attentions if output_attentions else None
        return hidden, attns, inputs.get("input_ids")

    # ── Caption generation ────────────────────────────────────────────────────

    def generate_caption(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 200,
    ) -> str:
        """
        Generate a textual caption for the given image.

        Args:
            image: PIL image
            prompt: captioning instruction (defaults to model config)
            max_new_tokens: max tokens to generate
        Returns:
            Caption string.
        """
        cap_prompt = prompt or self._caption_prompt
        inputs = self._prepare_vl(cap_prompt, image)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = generated[0][input_len:]
        caption = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return caption

    # ── Memory management ─────────────────────────────────────────────────────

    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
