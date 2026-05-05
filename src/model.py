"""
VLM wrappers for activation and attention extraction experiments.

Supports multiple architectures under a common interface:
  - LLaVA 1.5 / 1.6                 (LLaVAWrapper)
  - ShareGPT4V-7B                   (ShareGPT4VWrapper)
  - MiniGPT-4                       (MiniGPT4Wrapper)  [requires external repo]
  - InternVL2 / InternVL2.5-MPO     (InternVL2Wrapper)
  - Qwen-VL / Qwen-VL-Chat          (QwenVLWrapper)
  - Qwen2-VL / Qwen2.5-VL           (Qwen2VLWrapper)

All wrappers expose the same interface:
  - load()
  - forward_vl(image, text)  -> (hidden_states, attentions, input_ids)
  - forward_text(text)       -> (hidden_states, attentions, input_ids)
  - generate_vl(image, text, max_new_tokens)   -> str
  - generate_text(text, max_new_tokens)        -> str
  - generate_caption(image, max_new_tokens)    -> str
  - num_layers, hidden_dim, model_name, device (properties)

Usage
-----
    from src.model import create_wrapper
    wrapper = create_wrapper("llava-hf/llava-1.5-7b-hf").load()
    hidden, attn, ids = wrapper.forward_vl(image, text)

`VLMWrapper(model_id)` is kept as a backward-compatible factory function.
"""

import gc
import os
import re
import tempfile
import torch
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image


_DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


# ── Model name normalization ────────────────────────────────────────────────

def _normalize_model_name(model_id: str) -> str:
    """
    Normalize a HuggingFace model_id to a lowercase directory-safe name.

    Examples:
        llava-hf/llava-1.5-7b-hf           -> llava-1.5-7b-hf
        OpenGVLab/InternVL2-8B             -> internvl2-8b
        OpenGVLab/InternVL2_5-8B-MPO       -> internvl2.5-8b-mpo
        Qwen/Qwen2.5-VL-7B-Instruct        -> qwen2.5-vl-7b-instruct
    """
    name = model_id.split("/")[-1].lower()
    # InternVL uses underscores for dotted versions (InternVL2_5 == InternVL2.5)
    name = re.sub(r"(\d)_(\d)", r"\1.\2", name)
    return name


# ── Base class ──────────────────────────────────────────────────────────────

class VLMWrapperBase:
    """Abstract base class for multimodal model wrappers."""

    def __init__(
        self,
        model_id: str,
        torch_dtype=torch.float16,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.tokenizer = None

    # ── Common properties ──────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return _normalize_model_name(self.model_id)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    @property
    def hidden_dim(self) -> int:
        raise NotImplementedError

    # ── Required subclass interface ────────────────────────────────────────
    def load(self) -> "VLMWrapperBase":
        raise NotImplementedError

    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        raise NotImplementedError

    def forward_text(self, text: str, output_attentions: bool = False):
        raise NotImplementedError

    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        raise NotImplementedError

    def generate_captions_batch(self, images: List[Image.Image],
                                max_new_tokens: int = 200) -> List[str]:
        # Default: sequential fallback. Subclasses can override for efficiency.
        return [self.generate_caption(img, max_new_tokens=max_new_tokens)
                for img in images]

    # ── Memory management ──────────────────────────────────────────────────
    def cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── GPU diagnostics ────────────────────────────────────────────────────
    def _print_diagnostics(self):
        print(f"\n  CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device       : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM total        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM allocated    : {allocated:.1f} GB")
        primary = next(self.model.parameters()).device
        print(f"  Model device      : {primary}")
        if str(primary) == "cpu":
            print("  *** WARNING: Model is on CPU! Check CUDA/PyTorch installation. ***")
        print(f"  dtype             : {self.torch_dtype}")
        print(f"  num_hidden_layers : {self.num_layers}")
        print(f"  hidden_size       : {self.hidden_dim}")


# ── LLaVA wrapper (1.5 / 1.6) ───────────────────────────────────────────────

_LLAVA_CONFIGS = {
    "llava-hf/llava-1.5-7b-hf": {
        "model_class": "LlavaForConditionalGeneration",
        "num_image_tokens": 576,
        "prompt_template": "USER: <image>\n{text}\nASSISTANT:",
        "text_only_template": "USER: {text}\nASSISTANT:",
        "caption_prompt": "Describe this image in detail.",
    },
    "llava-hf/llava-v1.6-vicuna-7b-hf": {
        "model_class": "LlavaNextForConditionalGeneration",
        "num_image_tokens": None,
        "prompt_template": "USER: <image>\n{text}\nASSISTANT:",
        "text_only_template": "USER: {text}\nASSISTANT:",
        "caption_prompt": "Describe this image in detail.",
    },
}


class LLaVAWrapper(VLMWrapperBase):
    """Wrapper for LLaVA 1.5 and 1.6 models."""

    def __init__(self, model_id: str = _DEFAULT_MODEL, **kwargs):
        super().__init__(model_id, **kwargs)
        cfg = _LLAVA_CONFIGS.get(model_id, _LLAVA_CONFIGS[_DEFAULT_MODEL])
        self._model_class_name = cfg["model_class"]
        self._num_image_tokens_cfg = cfg["num_image_tokens"]
        self._prompt_template = cfg["prompt_template"]
        self._text_only_template = cfg["text_only_template"]
        self._caption_prompt = cfg["caption_prompt"]

    def load(self) -> "LLaVAWrapper":
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
        self._print_diagnostics()
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

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        return self.model.config.text_config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def num_attention_heads(self) -> int:
        return self.model.config.text_config.num_attention_heads

    @property
    def image_token_id(self) -> int:
        return self.model.config.image_token_index

    @property
    def num_image_tokens(self) -> int:
        if self._num_image_tokens_cfg is not None:
            return self._num_image_tokens_cfg
        vision_cfg = self.model.config.vision_config
        img_size = getattr(vision_cfg, "image_size", 336)
        patch_size = getattr(vision_cfg, "patch_size", 14)
        return (img_size // patch_size) ** 2

    # ── Image token position detection ────────────────────────────────────
    def get_image_token_span(self, input_ids: torch.Tensor) -> Tuple[Optional[int], Optional[int]]:
        img_id = self.image_token_id
        matches = (input_ids[0] == img_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            return None, None
        img_pos = int(matches[0].item())
        return img_pos, img_pos + self.num_image_tokens

    def get_text_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        img_start, img_end = self.get_image_token_span(input_ids)
        orig_len = input_ids.shape[1]
        if img_start is None:
            return list(range(orig_len))
        expanded_len = orig_len - 1 + self.num_image_tokens
        return (list(range(img_start)) + list(range(img_end, expanded_len)))

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_vl(self, text: str, image: Image.Image) -> dict:
        prompt = self._prompt_template.format(text=text)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _prepare_text(self, text: str) -> dict:
        prompt = self._text_only_template.format(text=text)
        inputs = self.processor(text=prompt, return_tensors="pt")
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        inputs = self._prepare_vl(text, image)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                inputs.get("input_ids"))

    def forward_text(self, text: str, output_attentions: bool = False):
        inputs = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                inputs.get("input_ids"))

    # ── Generation ─────────────────────────────────────────────────────────
    def generate_caption(self, image: Image.Image,
                         prompt: Optional[str] = None,
                         max_new_tokens: int = 200) -> str:
        cap_prompt = prompt or self._caption_prompt
        inputs = self._prepare_vl(cap_prompt, image)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        new_tokens = generated[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_captions_batch(self, images: List[Image.Image],
                                prompt: Optional[str] = None,
                                max_new_tokens: int = 200) -> List[str]:
        cap_prompt = prompt or self._caption_prompt
        prompts = [self._prompt_template.format(text=cap_prompt)] * len(images)
        inputs = self.processor(text=prompts, images=images,
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return [
            self.processor.decode(gen[input_len:], skip_special_tokens=True).strip()
            for gen in generated
        ]

    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        inputs = self._prepare_vl(text, image)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.processor.decode(
            generated[0][input_len:], skip_special_tokens=True).strip()

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        inputs = self._prepare_text(text)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.processor.decode(
            generated[0][input_len:], skip_special_tokens=True).strip()


# ── ShareGPT4V wrapper ─────────────────────────────────────────────────────

def _build_mm_projector(mm_hidden_size: int, hidden_size: int):
    """Build the 2-layer MLP projector used by LLaVA-1.5 / ShareGPT4V."""
    return torch.nn.Sequential(
        torch.nn.Linear(mm_hidden_size, hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size),
    )


class ShareGPT4VWrapper(VLMWrapperBase):
    """
    Wrapper for ShareGPT4V-7B (Lin-Chen/ShareGPT4V-7B).

    Architecturally identical to LLaVA-1.5 (CLIP ViT-L/14@336 + 2-layer MLP
    projector + Vicuna-7B).  The HuggingFace repo stores the LLM + projector
    weights in one checkpoint and references a separate CLIP vision tower repo.
    Since the repo has no auto-registration code, we load each component
    manually: LLaMA backbone, CLIP vision tower, and MLP projector.

    LLM backbone: Vicuna-7B (LLaMA-2), 32 layers, hidden_dim=4096.
    """

    _PROMPT_TEMPLATE = "USER: <image>\n{text}\nASSISTANT:"
    _TEXT_ONLY_TEMPLATE = "USER: {text}\nASSISTANT:"
    _CAPTION_PROMPT = "Describe this image in detail."

    def __init__(self, model_id: str = "Lin-Chen/ShareGPT4V-7B", **kwargs):
        super().__init__(model_id, **kwargs)
        self._image_processor = None
        self._vision_tower = None
        self._mm_projector = None
        self._mm_vision_select_layer = -2  # default: second-to-last ViT layer

    def load(self) -> "ShareGPT4VWrapper":
        import json
        from huggingface_hub import hf_hub_download
        from transformers import (
            AutoTokenizer, LlamaConfig, LlamaForCausalLM,
            CLIPVisionModel, CLIPImageProcessor,
        )

        # ── 1. Load tokenizer ────────────────────────────────────────
        print(f"Loading tokenizer from '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=False,
        )

        # ── 2. Read the original config to extract vision tower info ──
        cfg_path = hf_hub_download(self.model_id, "config.json")
        with open(cfg_path) as f:
            raw_cfg = json.load(f)
        vision_tower_id = raw_cfg.get(
            "mm_vision_tower",
            "openai/clip-vit-large-patch14-336",
        )
        mm_hidden_size = raw_cfg.get("mm_hidden_size", 1024)
        self._mm_vision_select_layer = raw_cfg.get("mm_vision_select_layer", -2)

        # ── 3. Load LLaMA backbone ────────────────────────────────────
        print(f"Loading LLM backbone (LlamaForCausalLM) from '{self.model_id}'...")
        llama_config = LlamaConfig.from_pretrained(self.model_id)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_id,
            config=llama_config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()

        # ── 4. Load CLIP vision tower ─────────────────────────────────
        print(f"Loading vision tower from '{vision_tower_id}'...")
        self._vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_id, torch_dtype=self.torch_dtype,
        )
        self._vision_tower.eval()
        self._vision_tower.to(self.device)
        self._image_processor = CLIPImageProcessor.from_pretrained(vision_tower_id)

        # ── 5. Build and load MLP projector ───────────────────────────
        print("Loading multimodal projector weights...")
        self._mm_projector = _build_mm_projector(mm_hidden_size, self.hidden_dim)
        # Load projector weights from the checkpoint
        from huggingface_hub import hf_hub_download
        idx_path = hf_hub_download(self.model_id, "pytorch_model.bin.index.json")
        with open(idx_path) as f:
            index = json.load(f)
        # Find which shard(s) contain mm_projector weights
        proj_files = set()
        proj_key_map = {}
        for key, shard in index["weight_map"].items():
            if "mm_projector" in key:
                proj_files.add(shard)
                # Map from checkpoint key to projector key
                # e.g. "model.mm_projector.0.weight" -> "0.weight"
                proj_key = key.replace("model.mm_projector.", "")
                proj_key_map[key] = proj_key
        proj_state = {}
        for shard_name in proj_files:
            shard_path = hf_hub_download(self.model_id, shard_name)
            shard_weights = torch.load(shard_path, map_location="cpu",
                                       weights_only=True)
            for full_key, local_key in proj_key_map.items():
                if full_key in shard_weights:
                    proj_state[local_key] = shard_weights[full_key]
        self._mm_projector.load_state_dict(proj_state)
        self._mm_projector.to(device=self.device, dtype=self.torch_dtype)
        self._mm_projector.eval()

        self._print_diagnostics()
        n_vis = (self._image_processor.size.get("height", 336) // 14) ** 2
        print(f"  vision_tower      : {vision_tower_id}")
        print(f"  num_visual_tokens : {n_vis}")
        return self

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    # ── Image preprocessing ────────────────────────────────────────────────
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image through the CLIP vision tower + MLP projector."""
        pixel_values = self._image_processor(
            image, return_tensors="pt",
        )["pixel_values"].to(device=self.device, dtype=self.torch_dtype)
        # Extract features from the selected ViT layer
        vit_out = self._vision_tower(pixel_values, output_hidden_states=True)
        image_features = vit_out.hidden_states[self._mm_vision_select_layer]
        # Remove CLS token (LLaVA-1.5 uses "patch" features, not CLS)
        image_features = image_features[:, 1:, :]
        # Project through the 2-layer MLP
        image_features = self._mm_projector(image_features)
        return image_features  # (1, n_patches, hidden_dim)

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_vl_embeds(self, image: Image.Image, text: str):
        """Build inputs_embeds with visual tokens spliced into the prompt.

        Vicuna-7B's stock tokenizer does NOT have `<image>` as a single
        special token; it tokenizes the literal string into BPE subword
        pieces, so we cannot find a single placeholder id and overwrite
        it. Instead we split the prompt on the placeholder, tokenize
        each half independently, and concatenate
            [before_embeds, image_features, after_embeds]
        — the same pattern `MiniGPT4Wrapper._prepare_vl_embeds` uses.
        """
        prompt = self._PROMPT_TEMPLATE.format(text=text)
        image_features = self._encode_image(image)  # (1, n_vis, dim)
        embed_fn = self.model.get_input_embeddings()

        if "<image>" not in prompt:
            # Defensive fallback: nothing to splice. Embed the whole prompt.
            input_ids = self.tokenizer(
                prompt, return_tensors="pt",
            ).input_ids.to(self.device)
            return embed_fn(input_ids), input_ids

        before_text, after_text = prompt.split("<image>", 1)
        before_ids = self.tokenizer(
            before_text, return_tensors="pt", add_special_tokens=True,
        ).input_ids.to(self.device)
        after_ids = self.tokenizer(
            after_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)

        before_embeds = embed_fn(before_ids)
        after_embeds = embed_fn(after_ids)
        image_features = image_features.to(dtype=before_embeds.dtype,
                                           device=before_embeds.device)

        inputs_embeds = torch.cat(
            [before_embeds, image_features, after_embeds], dim=1,
        )
        expanded_len = inputs_embeds.shape[1]
        input_ids = torch.zeros(1, expanded_len, dtype=torch.long,
                                device=self.device)
        return inputs_embeds, input_ids

    def _prepare_text(self, text: str):
        prompt = self._TEXT_ONLY_TEMPLATE.format(text=text)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt",
        ).input_ids.to(self.device)
        return input_ids

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        inputs_embeds, input_ids = self._prepare_vl_embeds(image, text)
        with torch.no_grad():
            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    def forward_text(self, text: str, output_attentions: bool = False):
        input_ids = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    # ── Generation ─────────────────────────────────────────────────────────
    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        inputs_embeds, _ = self._prepare_vl_embeds(image, text)
        with torch.no_grad():
            generated = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        input_ids = self._prepare_text(text)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.tokenizer.decode(
            generated[0][input_len:], skip_special_tokens=True,
        ).strip()

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        return self.generate_vl(image, self._CAPTION_PROMPT,
                                max_new_tokens=max_new_tokens)


# ── MiniGPT-4 wrapper ──────────────────────────────────────────────────────

class MiniGPT4Wrapper(VLMWrapperBase):
    """
    Wrapper for MiniGPT-4 (Vision-CAIR/MiniGPT-4).

    Architecture: BLIP-2 ViT-G/14 + Q-Former + single linear projection
    + Vicuna-7B (frozen).  LLM backbone: Vicuna-7B, 32 layers, hidden_dim=4096.

    REQUIRES the MiniGPT-4 repository to be installed:
      1. git clone https://github.com/Vision-CAIR/MiniGPT-4.git
      2. Add the repo root to PYTHONPATH
      3. Download the pretrained checkpoint and either:
         - Set the MINIGPT4_CKPT environment variable, or
         - Pass ckpt_path= to the constructor

    The wrapper loads the model via MiniGPT-4's own registry and config
    system, then extracts hidden states from the underlying LLaMA backbone.
    """

    _PROMPT_TEMPLATE = "###Human: {} ###Assistant: "
    _TEXT_ONLY_TEMPLATE = "###Human: {} ###Assistant: "
    _CAPTION_PROMPT = "Describe this image in detail."
    _END_SYM = "###"

    def __init__(self, model_id: str = "Vision-CAIR/MiniGPT-4", **kwargs):
        self._ckpt_path = kwargs.pop("ckpt_path", None)
        self._cfg_path = kwargs.pop("cfg_path", None)
        kwargs.setdefault("torch_dtype", torch.float16)
        super().__init__(model_id, **kwargs)
        self._vis_processor = None

    def load(self) -> "MiniGPT4Wrapper":
        # ── Import guard ──────────────────────────────────────────────
        try:
            from minigpt4.common.registry import registry
            import minigpt4.models.minigpt4  # noqa: F401  — registers the model
        except ImportError as e:
            raise ImportError(
                "MiniGPT-4 requires the Vision-CAIR/MiniGPT-4 repository.\n"
                "  1. git clone https://github.com/Vision-CAIR/MiniGPT-4.git\n"
                "  2. Add the repo root to PYTHONPATH:\n"
                "       export PYTHONPATH=/path/to/MiniGPT-4:$PYTHONPATH\n"
                "  3. Download the pretrained checkpoint and set:\n"
                "       export MINIGPT4_CKPT=/path/to/pretrained_minigpt4_7b.pth\n"
                f"  Original error: {e}"
            ) from e

        from omegaconf import OmegaConf

        ckpt_path = self._ckpt_path or os.environ.get("MINIGPT4_CKPT", "")
        if not ckpt_path:
            raise ValueError(
                "MiniGPT-4 checkpoint not specified. Set MINIGPT4_CKPT env var "
                "or pass ckpt_path= to the constructor."
            )

        print(f"Loading MiniGPT-4 from checkpoint '{ckpt_path}'...")

        # Build a minimal config programmatically
        model_cfg = OmegaConf.create({
            "arch": "minigpt4",
            "model_type": "pretrain_vicuna0",
            "max_txt_len": 160,
            "end_sym": self._END_SYM,
            "low_resource": True,
            "prompt_template": self._PROMPT_TEMPLATE,
            "ckpt": ckpt_path,
        })

        model_cls = registry.get_model_class("minigpt4")
        self.model = model_cls.from_config(model_cfg)
        self.model.eval()
        if torch.cuda.is_available() and not model_cfg.get("low_resource", False):
            self.model = self.model.cuda()

        # Set up vision preprocessor
        from torchvision import transforms
        self._vis_processor = transforms.Compose([
            transforms.Resize((224, 224),
                               interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.tokenizer = self.model.llama_tokenizer
        self._print_diagnostics()
        return self

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def device(self):
        return next(self.model.llama_model.parameters()).device

    @property
    def num_layers(self) -> int:
        return self.model.llama_model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.llama_model.config.hidden_size

    # ── Image preprocessing ────────────────────────────────────────────────
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Returns pixel tensor of shape (1, 3, 224, 224) on the model device."""
        image = image.convert("RGB")
        pixel_values = self._vis_processor(image).unsqueeze(0)
        return pixel_values.to(device=self.device, dtype=self.torch_dtype)

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_vl_embeds(self, image: Image.Image, text: str):
        """Encode image, build prompt, splice visual tokens into embeddings."""
        pixel_values = self._preprocess_image(image)
        # Encode image through ViT + Q-Former + projection
        with torch.no_grad():
            image_embeds, _ = self.model.encode_img(pixel_values)

        # Build prompt with <ImageHere> placeholder
        prompt = self._PROMPT_TEMPLATE.format(
            "<Img><ImageHere></Img> " + text
        )

        # Tokenize and build embeddings
        parts = prompt.split("<ImageHere>")
        before_ids = self.tokenizer(
            parts[0], return_tensors="pt", add_special_tokens=True,
        ).input_ids.to(self.device)
        after_ids = self.tokenizer(
            parts[1], return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)

        embed_fn = self.model.llama_model.get_input_embeddings()
        before_embeds = embed_fn(before_ids)
        after_embeds = embed_fn(after_ids)

        # Concatenate: [text_before] [visual_tokens] [text_after]
        inputs_embeds = torch.cat(
            [before_embeds, image_embeds, after_embeds], dim=1,
        )
        # Build a dummy input_ids tensor for return (actual tokens not meaningful
        # after splice, but needed for shape tracking in downstream code)
        total_len = inputs_embeds.shape[1]
        input_ids = torch.zeros(1, total_len, dtype=torch.long,
                                device=self.device)
        return inputs_embeds, input_ids

    def _prepare_text(self, text: str):
        prompt = self._TEXT_ONLY_TEMPLATE.format(text)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        ).input_ids.to(self.device)
        return input_ids

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        inputs_embeds, input_ids = self._prepare_vl_embeds(image, text)
        with torch.no_grad():
            outputs = self.model.llama_model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    def forward_text(self, text: str, output_attentions: bool = False):
        input_ids = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model.llama_model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    # ── Generation ─────────────────────────────────────────────────────────
    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        inputs_embeds, _ = self._prepare_vl_embeds(image, text)
        with torch.no_grad():
            generated = self.model.llama_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        # Strip the end symbol
        if self._END_SYM in output:
            output = output[:output.index(self._END_SYM)]
        return output.strip()

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        input_ids = self._prepare_text(text)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            generated = self.model.llama_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        output = self.tokenizer.decode(
            generated[0][input_len:], skip_special_tokens=True,
        )
        if self._END_SYM in output:
            output = output[:output.index(self._END_SYM)]
        return output.strip()

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        return self.generate_vl(image, self._CAPTION_PROMPT,
                                max_new_tokens=max_new_tokens)


# ── InternVL2 / InternVL2.5 wrapper ─────────────────────────────────────────

# InternVL preprocessing constants (matches OpenGVLab/InternVL2-8B README)
_INTERNVL_IMG_SIZE = 448
_INTERNVL_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_STD = (0.229, 0.224, 0.225)


def _build_internvl_transform(input_size: int = _INTERNVL_IMG_SIZE):
    """Build the InternVL2 image preprocessing pipeline."""
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    from torchvision.transforms.functional import InterpolationMode
    return Compose([
        lambda img: img.convert("RGB") if img.mode != "RGB" else img,
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=_INTERNVL_MEAN, std=_INTERNVL_STD),
    ])


class InternVL2Wrapper(VLMWrapperBase):
    """
    Wrapper for InternVL2-8B and InternVL2.5-8B-MPO.

    Both share the same architecture (ViT-MLP-LLM with InternLM2.5-7B-Chat
    backbone, 32 layers, hidden_dim=4096) and API.

    NOTE: trust_remote_code=True is required — model code is downloaded
    from HuggingFace. Dynamic image tiling is simplified here to a single
    448x448 tile (max_num=1) to keep the visual token count deterministic
    (256 tokens from the 16x16 patch grid).
    """

    def __init__(self, model_id: str, **kwargs):
        # Default to bfloat16 for InternVL2 per the HF model card
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        super().__init__(model_id, **kwargs)
        self._transform = None
        self._img_context_token_id = None
        self._num_image_token = None  # visual tokens per tile
        self._IMG_START = "<img>"
        self._IMG_END = "</img>"
        self._IMG_CONTEXT = "<IMG_CONTEXT>"

    def load(self) -> "InternVL2Wrapper":
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading tokenizer from '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False,
        )

        print(f"Loading model (AutoModel) from '{self.model_id}'...")
        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # device_map: "auto" can split across GPUs; for most cases a single
        # GPU works fine. Default to cuda if available, else cpu.
        try:
            self.model = AutoModel.from_pretrained(
                self.model_id, device_map=self.device_map, **load_kwargs,
            )
        except (RuntimeError, AttributeError):
            # Some models (e.g. InternVL2) have two incompatibilities with
            # transformers 5.x:
            # 1. .item() calls during __init__ fail on meta tensors
            #    (transformers 5.x always inits on meta device).
            # 2. Missing `all_tied_weights_keys` attr expected by
            #    _finalize_model_loading (custom code targets older API).
            # Fix both by monkey-patching during loading.
            from transformers import PreTrainedModel
            _orig_ctx = PreTrainedModel.get_init_context
            @classmethod
            def _no_meta_get_init_context(cls, *args, **kwargs):
                ctxs = _orig_ctx.__func__(cls, *args, **kwargs)
                return [c for c in ctxs if c != torch.device("meta")]
            _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
            def _safe_mark(self_inner, loading_info):
                if not hasattr(self_inner, "all_tied_weights_keys"):
                    self_inner.all_tied_weights_keys = {}
                return _orig_mark(self_inner, loading_info)
            PreTrainedModel.get_init_context = _no_meta_get_init_context
            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_id, **load_kwargs,
                )
            finally:
                PreTrainedModel.get_init_context = _orig_ctx
                PreTrainedModel.mark_tied_weights_as_initialized = _orig_mark
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        self.model.eval()

        # Resolve IMG_CONTEXT token id (used to substitute visual embeddings)
        self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(self._IMG_CONTEXT)
        if hasattr(self.model, "img_context_token_id"):
            self.model.img_context_token_id = self._img_context_token_id

        # Number of visual tokens per tile
        self._num_image_token = int(getattr(self.model, "num_image_token", 256))
        self._transform = _build_internvl_transform()
        self._print_diagnostics()
        print(f"  num_image_tokens  : {self._num_image_token}")
        print(f"  img_context_token : {self._IMG_CONTEXT} (id={self._img_context_token_id})")
        return self

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        # LLM submodule holds the transformer layers
        return self.model.language_model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.language_model.config.hidden_size

    @property
    def num_image_tokens(self) -> int:
        return self._num_image_token

    # ── Image preprocessing ────────────────────────────────────────────────
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Returns pixel_values of shape (1, 3, H, W) on the model's device."""
        pixel_values = self._transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(device=self.device, dtype=self.torch_dtype)
        return pixel_values

    def _build_vl_prompt(self, text: str) -> str:
        """Build a single-turn InternLM2-chat prompt with image placeholder."""
        img_block = self._IMG_START + self._IMG_CONTEXT * self._num_image_token + self._IMG_END
        # InternLM2 chat template
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{img_block}\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_text_prompt(self, text: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_inputs_embeds(self, input_ids: torch.Tensor,
                               pixel_values: Optional[torch.Tensor]):
        """Embed input_ids and splice visual features into IMG_CONTEXT positions."""
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            vit_embeds = self.model.extract_feature(pixel_values)  # (B*N, n_tok, dim)
            vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
            mask = (input_ids == self._img_context_token_id)
            flat = input_embeds.reshape(-1, input_embeds.shape[-1])
            flat_mask = mask.reshape(-1)
            if int(flat_mask.sum()) != vit_embeds.shape[0]:
                raise RuntimeError(
                    f"IMG_CONTEXT token count ({int(flat_mask.sum())}) does not "
                    f"match visual tokens produced ({vit_embeds.shape[0]})"
                )
            flat[flat_mask] = vit_embeds.to(flat.dtype)
            input_embeds = flat.reshape(input_embeds.shape)
        return input_embeds

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        prompt = self._build_vl_prompt(text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        pixel_values = self._preprocess_image(image)
        input_embeds = self._prepare_inputs_embeds(input_ids, pixel_values)
        with torch.no_grad():
            outputs = self.model.language_model(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    def forward_text(self, text: str, output_attentions: bool = False):
        prompt = self._build_text_prompt(text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.language_model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    # ── Generation ─────────────────────────────────────────────────────────
    def _generate(self, input_ids: torch.Tensor,
                  pixel_values: Optional[torch.Tensor],
                  max_new_tokens: int) -> str:
        input_embeds = self._prepare_inputs_embeds(input_ids, pixel_values)
        with torch.no_grad():
            generated = self.model.language_model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                use_cache=True,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        return self.generate_vl(image, "Describe this image in detail.",
                                max_new_tokens=max_new_tokens)

    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        prompt = self._build_vl_prompt(text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        pixel_values = self._preprocess_image(image)
        return self._generate(input_ids, pixel_values, max_new_tokens)

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        prompt = self._build_text_prompt(text)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        return self._generate(input_ids, None, max_new_tokens)


# ── Qwen-VL (original) wrapper ─────────────────────────────────────────────

class QwenVLWrapper(VLMWrapperBase):
    """
    Wrapper for the original Qwen-VL / Qwen-VL-Chat (Qwen/Qwen-VL-Chat).

    Architecture: Qwen-7B backbone with integrated vision encoder.
    Fixed 448x448 input resolution.  Requires trust_remote_code=True.

    LLM backbone: Qwen-7B, 32 layers, hidden_dim=4096.

    NOTE: This is distinct from Qwen2-VL / Qwen2.5-VL, which use a different
    architecture and processor.  The original Qwen-VL uses a custom tokenizer
    with from_list_format() for image handling and expects image file paths
    (not PIL images), so we save to a temp file for each VL call.
    """

    _CAPTION_PROMPT = "Describe this image in detail."

    def __init__(self, model_id: str = "Qwen/Qwen-VL-Chat", **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        super().__init__(model_id, **kwargs)

    def load(self) -> "QwenVLWrapper":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        print(f"Loading tokenizer from '{self.model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True,
        )

        print(f"Loading model (Qwen-VL) from '{self.model_id}'...")
        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        # On GPUs with < 20 GiB VRAM, Qwen-VL (~19 GiB in bf16) needs
        # CPU offloading.  But Qwen-VL's trust_remote_code visual encoder
        # bypasses accelerate's CPU-offload hooks, so any weight that
        # accelerate puts on `meta` and tries to swap-in via a hook fails
        # with "Cannot copy out of meta tensor; no data!" during the
        # forward pass.  Workaround: build an explicit device_map that
        # pins `transformer.visual` (and the small embedding / output
        # head modules used on every step) to GPU, and lets accelerate
        # CPU-offload only the standard LLM transformer layers, which
        # it can hook correctly.
        if self.device_map == "auto" and torch.cuda.is_available():
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gib < 20:
                from accelerate import init_empty_weights, infer_auto_device_map
                config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
                with init_empty_weights():
                    empty_model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True,
                        torch_dtype=self.torch_dtype,
                    )
                no_split = getattr(empty_model, "_no_split_modules", None) or []
                _pin_to_gpu_prefixes = (
                    "transformer.visual",
                    "transformer.wte",
                    "transformer.ln_f",
                    "lm_head",
                )

                def _is_pinned(name: str) -> bool:
                    return any(name == p or name.startswith(p + ".")
                               for p in _pin_to_gpu_prefixes)

                # Size of the modules we'll pin to GPU.
                pinned_bytes = 0
                for name, module in empty_model.named_modules():
                    if not _is_pinned(name):
                        continue
                    # Only count direct params (named_modules walks the tree).
                    for p in module.parameters(recurse=False):
                        pinned_bytes += p.numel() * p.element_size()
                    for b in module.buffers(recurse=False):
                        pinned_bytes += b.numel() * b.element_size()
                pinned_gib = pinned_bytes / (1024 ** 3)

                # Reserve 5 GiB for forward-pass overhead, then split the
                # remaining VRAM between the pinned modules and the LLM
                # layers that infer_auto_device_map will distribute.
                headroom_gib = 5
                llm_budget_gib = max(int(total_vram_gib - headroom_gib - pinned_gib), 2)
                print(f"  GPU < 20 GiB detected ({total_vram_gib:.1f} GiB) — "
                      f"pinning vision encoder to GPU "
                      f"(pinned {pinned_gib:.1f} GiB), capping LLM "
                      f"GPU budget to {llm_budget_gib} GiB")
                device_map = infer_auto_device_map(
                    empty_model,
                    max_memory={0: f"{llm_budget_gib}GiB", "cpu": "30GiB"},
                    no_split_module_classes=no_split,
                    dtype=self.torch_dtype,
                )
                for name in list(device_map.keys()):
                    if _is_pinned(name):
                        device_map[name] = 0
                del empty_model
                load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self._print_diagnostics()
        return self

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    # ── Image helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _save_image_temp(image: Image.Image) -> str:
        """Save a PIL image to a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.convert("RGB").save(tmp, format="PNG")
        tmp.close()
        return tmp.name

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_vl(self, image: Image.Image, text: str):
        """Tokenize a VL prompt using Qwen-VL's from_list_format.

        Returns (input_ids, img_path). The caller MUST unlink img_path
        after the forward/generate call — Qwen-VL's tokenizer embeds the
        path in the input_ids and the image is loaded lazily during the
        model's forward pass, so deleting the file earlier raises
        FileNotFoundError inside the model.
        """
        img_path = self._save_image_temp(image)
        query = self.tokenizer.from_list_format([
            {"image": img_path},
            {"text": text},
        ])
        input_ids = self.tokenizer(
            query, return_tensors="pt",
        ).input_ids.to(self.device)
        return input_ids, img_path

    def _prepare_text(self, text: str):
        input_ids = self.tokenizer(
            text, return_tensors="pt",
        ).input_ids.to(self.device)
        return input_ids

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        input_ids, img_path = self._prepare_vl(image, text)
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=False,
                )
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    def forward_text(self, text: str, output_attentions: bool = False):
        input_ids = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                input_ids)

    # ── Generation ─────────────────────────────────────────────────────────
    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        img_path = self._save_image_temp(image)
        query = self.tokenizer.from_list_format([
            {"image": img_path},
            {"text": text},
        ])
        gen_config = getattr(self.model, "generation_config", None)
        old_max_new_tokens = getattr(gen_config, "max_new_tokens", None)
        old_do_sample = getattr(gen_config, "do_sample", None)
        try:
            # Qwen-VL-Chat expects its chat API for generation; raw generate()
            # behaves like continuation and can produce OCR-like repetition.
            if gen_config is not None:
                gen_config.max_new_tokens = max_new_tokens
                gen_config.do_sample = False
            with torch.no_grad():
                response, _ = self.model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                )
        finally:
            if gen_config is not None:
                if old_max_new_tokens is None:
                    try:
                        delattr(gen_config, "max_new_tokens")
                    except AttributeError:
                        pass
                else:
                    gen_config.max_new_tokens = old_max_new_tokens
                if old_do_sample is None:
                    try:
                        delattr(gen_config, "do_sample")
                    except AttributeError:
                        pass
                else:
                    gen_config.do_sample = old_do_sample
            try:
                os.unlink(img_path)
            except OSError:
                pass
        return response.strip()

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        input_ids = self._prepare_text(text)
        input_len = input_ids.shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.tokenizer.decode(
            generated[0][input_len:], skip_special_tokens=True,
        ).strip()

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        return self.generate_vl(image, self._CAPTION_PROMPT,
                                max_new_tokens=max_new_tokens)


# ── Qwen2-VL / Qwen2.5-VL wrapper ───────────────────────────────────────────

_QWEN_VISION_START = "<|vision_start|>"
_QWEN_VISION_END = "<|vision_end|>"
_QWEN_IMAGE_PAD = "<|image_pad|>"

_QWEN_FALLBACK_VL_TEMPLATE = (
    f"USER: {_QWEN_VISION_START}{_QWEN_IMAGE_PAD}{_QWEN_VISION_END}"
    "{text}\nASSISTANT:"
)
_QWEN_FALLBACK_TEXT_TEMPLATE = "USER: {text}\nASSISTANT:"

_QWEN_CONFIGS = {
    "Qwen/Qwen2-VL-7B": {
        "model_class": "Qwen2VLForConditionalGeneration",
        "use_chat_template": True,
        "fallback_vl_template": _QWEN_FALLBACK_VL_TEMPLATE,
        "fallback_text_template": _QWEN_FALLBACK_TEXT_TEMPLATE,
        "caption_prompt": "Describe this image in detail.",
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "model_class": "Qwen2VLForConditionalGeneration",
        "use_chat_template": True,
        "fallback_vl_template": _QWEN_FALLBACK_VL_TEMPLATE,
        "fallback_text_template": _QWEN_FALLBACK_TEXT_TEMPLATE,
        "caption_prompt": "Describe this image in detail.",
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "model_class": "Qwen2_5_VLForConditionalGeneration",
        "use_chat_template": True,
        "fallback_vl_template": _QWEN_FALLBACK_VL_TEMPLATE,
        "fallback_text_template": _QWEN_FALLBACK_TEXT_TEMPLATE,
        "caption_prompt": "Describe this image in detail.",
    },
}

# Picked when an unknown Qwen model_id is passed.
_QWEN_DEFAULT_CFG = _QWEN_CONFIGS["Qwen/Qwen2-VL-7B-Instruct"]


class Qwen2VLWrapper(VLMWrapperBase):
    """
    Wrapper for Qwen2-VL and Qwen2.5-VL families.

    Supported:
      - Qwen/Qwen2-VL-7B            (base, no SFT — chat template falls back
                                     to a USER/ASSISTANT completion prompt)
      - Qwen/Qwen2-VL-7B-Instruct
      - Qwen/Qwen2.5-VL-7B-Instruct

    LLM backbone: Qwen2(.5)-7B, 28 layers, hidden_dim=3584.
    Visual tokens use a dynamic resolution scheme; the processor expands
    the <|image_pad|> marker into the correct number of vision tokens.
    """

    def __init__(self, model_id: str, max_pixels: Optional[int] = None, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        super().__init__(model_id, **kwargs)
        cfg = _QWEN_CONFIGS.get(model_id, _QWEN_DEFAULT_CFG)
        self._model_class_name = cfg["model_class"]
        self._use_chat_template = cfg["use_chat_template"]
        self._fallback_vl_template = cfg["fallback_vl_template"]
        self._fallback_text_template = cfg["fallback_text_template"]
        self._caption_prompt = cfg["caption_prompt"]
        # Cap vision tokens to fit in constrained VRAM. Default 512*28*28
        # = 401,408 pixels ≈ 512 vision tokens (comparable to LLaVA's 576).
        # Pass None to use the processor's built-in default (~1280 tokens).
        self._max_pixels = max_pixels if max_pixels is not None else 512 * 28 * 28
        # None until first _build_prompt() call — then True if chat template
        # works for this checkpoint, False if we permanently fell back.
        self._chat_template_ok: Optional[bool] = None

    def load(self) -> "Qwen2VLWrapper":
        from transformers import AutoProcessor
        print(f"Loading processor from '{self.model_id}'...")
        proc_kwargs = {}
        if self._max_pixels is not None:
            proc_kwargs["max_pixels"] = self._max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kwargs)

        print(f"Loading model ({self._model_class_name}) from '{self.model_id}'...")
        ModelClass = self._get_model_class()
        load_kwargs = dict(
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        # On GPUs with < 20 GiB VRAM, the full model (~16.6 GiB in bf16)
        # doesn't fit.  device_map="auto" will offload layers to CPU, but
        # without a memory budget it packs the GPU too tightly, leaving no
        # room for the layer-swap + forward-pass overhead (~4 GiB for the
        # vision encoder intermediates, accumulated hidden states, and one
        # offloaded transformer layer being swapped in at ~458 MiB each).
        # Reserve 5 GiB of headroom.
        if self.device_map == "auto" and torch.cuda.is_available():
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gib < 20:
                gpu_budget = f"{int(total_vram_gib) - 5}GiB"
                load_kwargs["max_memory"] = {0: gpu_budget, "cpu": "24GiB"}
                print(f"  GPU < 20 GiB detected ({total_vram_gib:.1f} GiB) — "
                      f"capping GPU budget to {gpu_budget}")
        self.model = ModelClass.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self._print_diagnostics()
        effective_max = getattr(self.processor.image_processor, "max_pixels", "unknown")
        print(f"  max_pixels        : {effective_max}")
        return self

    def _get_model_class(self):
        name = self._model_class_name
        if name == "Qwen2_5_VLForConditionalGeneration":
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                return Qwen2_5_VLForConditionalGeneration
            except ImportError as e:
                raise ImportError(
                    "Qwen2.5-VL requires transformers>=4.49. "
                    f"Original error: {e}"
                ) from e
        if name == "Qwen2VLForConditionalGeneration":
            try:
                from transformers import Qwen2VLForConditionalGeneration
                return Qwen2VLForConditionalGeneration
            except ImportError as e:
                raise ImportError(
                    "Qwen2-VL requires transformers>=4.45. "
                    f"Original error: {e}"
                ) from e
        raise ValueError(f"Unknown Qwen model class: {name}")

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        cfg = self.model.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            return cfg.text_config.num_hidden_layers
        return cfg.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        cfg = self.model.config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            return cfg.text_config.hidden_size
        return cfg.hidden_size

    # ── Image/text token position helpers ──────────────────────────────────
    def _token_id(self, token: str) -> Optional[int]:
        tid = self.processor.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.processor.tokenizer.unk_token_id:
            return None
        return tid

    def get_image_token_span(
        self, input_ids: torch.Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        vs_id = self._token_id(_QWEN_VISION_START)
        ve_id = self._token_id(_QWEN_VISION_END)
        if vs_id is None or ve_id is None:
            return None, None
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        starts = (row == vs_id).nonzero(as_tuple=True)[0]
        ends = (row == ve_id).nonzero(as_tuple=True)[0]
        if len(starts) == 0 or len(ends) == 0:
            return None, None
        # Span of the visual tokens between the markers (exclusive of markers).
        return int(starts[0].item()) + 1, int(ends[0].item())

    def get_text_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        vs_id = self._token_id(_QWEN_VISION_START)
        ve_id = self._token_id(_QWEN_VISION_END)
        pad_id = self._token_id(_QWEN_IMAGE_PAD)
        vision_ids = {i for i in (vs_id, ve_id, pad_id) if i is not None}
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        return [i for i, tok in enumerate(row.tolist()) if tok not in vision_ids]

    # ── Input preparation ──────────────────────────────────────────────────
    def _build_messages_vl(self, image: Image.Image, text: str):
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]

    def _build_messages_text(self, text: str):
        return [{
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }]

    def _build_prompt(self, kind: str, text: str,
                      image: Optional[Image.Image] = None) -> str:
        """Return a prompt string. Tries the chat template when enabled; on
        failure (e.g. missing template on a base checkpoint) falls back to a
        USER/ASSISTANT completion template and caches the decision."""
        if self._use_chat_template and self._chat_template_ok is not False:
            try:
                if kind == "vl":
                    messages = self._build_messages_vl(image, text)
                else:
                    messages = self._build_messages_text(text)
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                if prompt:
                    if self._chat_template_ok is None:
                        self._chat_template_ok = True
                    return prompt
            except Exception as e:
                if self._chat_template_ok is None:
                    print(
                        f"[Qwen2VLWrapper] chat template unavailable for "
                        f"'{self.model_id}' ({type(e).__name__}: {e}) — "
                        f"falling back to completion template."
                    )
                self._chat_template_ok = False
            else:
                if self._chat_template_ok is None:
                    print(
                        f"[Qwen2VLWrapper] chat template returned empty for "
                        f"'{self.model_id}' — falling back to completion "
                        f"template."
                    )
                self._chat_template_ok = False

        if kind == "vl":
            return self._fallback_vl_template.format(text=text)
        return self._fallback_text_template.format(text=text)

    def _prepare_vl(self, image: Image.Image, text: str) -> dict:
        prompt = self._build_prompt("vl", text, image=image)
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _prepare_text(self, text: str) -> dict:
        prompt = self._build_prompt("text", text)
        inputs = self.processor(
            text=[prompt], return_tensors="pt", padding=True,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    # ── Forward passes ─────────────────────────────────────────────────────
    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        inputs = self._prepare_vl(image, text)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                inputs.get("input_ids"))

    def forward_text(self, text: str, output_attentions: bool = False):
        inputs = self._prepare_text(text)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
            )
        return (outputs.hidden_states,
                outputs.attentions if output_attentions else None,
                inputs.get("input_ids"))

    # ── Generation ─────────────────────────────────────────────────────────
    def _generate_from_inputs(self, inputs: dict, max_new_tokens: int) -> str:
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        new_tokens = generated[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 256) -> str:
        return self._generate_from_inputs(
            self._prepare_vl(image, text), max_new_tokens)

    def generate_text(self, text: str, max_new_tokens: int = 256) -> str:
        return self._generate_from_inputs(
            self._prepare_text(text), max_new_tokens)

    def generate_caption(self, image: Image.Image,
                         max_new_tokens: int = 200) -> str:
        return self.generate_vl(image, self._caption_prompt,
                                max_new_tokens=max_new_tokens)


# ── Factory ─────────────────────────────────────────────────────────────────

def create_wrapper(model_id: str = _DEFAULT_MODEL, **kwargs) -> VLMWrapperBase:
    """Return the appropriate wrapper subclass for the given model_id."""
    m = model_id.lower()
    # Order matters: more-specific checks before generic family matches.
    if "sharegpt4v" in m or "share4v" in m:
        return ShareGPT4VWrapper(model_id, **kwargs)
    if "llava" in m:
        return LLaVAWrapper(model_id, **kwargs)
    if "minigpt" in m:
        return MiniGPT4Wrapper(model_id, **kwargs)
    if "internvl" in m:
        return InternVL2Wrapper(model_id, **kwargs)
    # Qwen2-VL / Qwen2.5-VL must match before generic "qwen" (original Qwen-VL)
    if "qwen2" in m:
        return Qwen2VLWrapper(model_id, **kwargs)
    if "qwen" in m:
        return QwenVLWrapper(model_id, **kwargs)
    raise ValueError(
        f"Unknown model: '{model_id}'. Supported families: "
        f"llava, sharegpt4v, minigpt4, internvl, qwen-vl, qwen2-vl."
    )


# Backward-compatible factory alias (existing code calls `VLMWrapper(args.model)`)
def VLMWrapper(model_id: str = _DEFAULT_MODEL, **kwargs) -> VLMWrapperBase:
    """Backward-compatible factory. Prefer `create_wrapper` in new code."""
    return create_wrapper(model_id, **kwargs)
