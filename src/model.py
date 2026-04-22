"""
VLM wrappers for activation and attention extraction experiments.

Supports multiple architectures under a common interface:
  - LLaVA 1.5 / 1.6                 (LLaVAWrapper)
  - InternVL2 / InternVL2.5-MPO     (InternVL2Wrapper)
  - Qwen2.5-VL                      (Qwen2VLWrapper)

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
import re
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

    @property
    def llm_layers(self):
        """Return the transformer decoder ModuleList for hook registration."""
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
    def llm_layers(self):
        # LlavaForConditionalGeneration → LlavaModel → LlamaModel → layers
        # Try model.model.language_model[.model].layers first, then
        # model.language_model[.model].layers for older checkpoints.
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "language_model"):
            lm = inner.language_model
            if hasattr(lm, "layers"):
                return lm.layers
            nested = getattr(lm, "model", None)
            if nested is not None and hasattr(nested, "layers"):
                return nested.layers
        lm = getattr(self.model, "language_model", None)
        if lm is not None:
            nested = getattr(lm, "model", None)
            if nested is not None and hasattr(nested, "layers"):
                return nested.layers
            if hasattr(lm, "layers"):
                return lm.layers
        raise AttributeError(
            "Could not resolve llm_layers for this LLaVA model."
        )

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
        return self.model.language_model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.language_model.config.hidden_size

    @property
    def num_image_tokens(self) -> int:
        return self._num_image_token

    @property
    def llm_layers(self):
        # InternVLChatModel → language_model (InternLM2ForCausalLM) → model → layers
        return self.model.language_model.model.layers

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


# ── Qwen2.5-VL wrapper ──────────────────────────────────────────────────────

class Qwen2VLWrapper(VLMWrapperBase):
    """
    Wrapper for Qwen2.5-VL-7B-Instruct.

    LLM backbone: Qwen2.5-7B, 28 layers, hidden_dim=3584.
    Visual tokens use a dynamic resolution scheme.
    """

    def __init__(self, model_id: str, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        super().__init__(model_id, **kwargs)

    def load(self) -> "Qwen2VLWrapper":
        from transformers import AutoProcessor
        print(f"Loading processor from '{self.model_id}'...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        print(f"Loading model from '{self.model_id}'...")
        ModelClass = self._get_model_class()
        self.model = ModelClass.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        self._print_diagnostics()
        return self

    def _get_model_class(self):
        # Try Qwen2_5_VL first (transformers >= 4.49), fall back to Qwen2VL
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            return Qwen2VLForConditionalGeneration

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        # Qwen2.5-VL config has a nested LLM config
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

    @property
    def llm_layers(self):
        # Qwen2_5_VLForConditionalGeneration → model (Qwen2_5_VLModel) → layers
        return self.model.model.layers

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

    def _prepare_vl(self, image: Image.Image, text: str) -> dict:
        messages = self._build_messages_vl(image, text)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()}

    def _prepare_text(self, text: str) -> dict:
        messages = self._build_messages_text(text)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
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
        return self.generate_vl(image, "Describe this image in detail.",
                                max_new_tokens=max_new_tokens)


# ── Factory ─────────────────────────────────────────────────────────────────

def create_wrapper(model_id: str = _DEFAULT_MODEL, **kwargs) -> VLMWrapperBase:
    """Return the appropriate wrapper subclass for the given model_id."""
    m = model_id.lower()
    if "llava" in m:
        return LLaVAWrapper(model_id, **kwargs)
    if "internvl" in m:
        return InternVL2Wrapper(model_id, **kwargs)
    if "qwen" in m:
        return Qwen2VLWrapper(model_id, **kwargs)
    raise ValueError(
        f"Unknown model: '{model_id}'. Supported families: llava, internvl, qwen."
    )


# Backward-compatible factory alias (existing code calls `VLMWrapper(args.model)`)
def VLMWrapper(model_id: str = _DEFAULT_MODEL, **kwargs) -> VLMWrapperBase:
    """Backward-compatible factory. Prefer `create_wrapper` in new code."""
    return create_wrapper(model_id, **kwargs)
