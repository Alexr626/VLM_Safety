"""
VLM wrappers for activation extraction and inference-time interventions.

Supported models:
  - llava-hf/llava-1.5-7b-hf         (LLaVAWrapper)
  - Qwen/Qwen2.5-VL-7B-Instruct      (Qwen2VLWrapper)

All wrappers expose the same interface:
  - load()
  - forward_vl(image, text)  -> (hidden_states, attentions, input_ids)
  - forward_text(text)       -> (hidden_states, attentions, input_ids)
  - generate_vl(image, text, max_new_tokens)   -> str
  - generate_text(text, max_new_tokens)        -> str
  - generate_caption(image, max_new_tokens)    -> str
  - prepare_vl(image, text) / prepare_text(text) -> processor inputs on device
  - num_layers, hidden_dim, model_name, device (properties)

Usage
-----
    from src.model import create_wrapper
    wrapper = create_wrapper("llava-hf/llava-1.5-7b-hf", pad_to_square=True).load()
    hidden, attn, ids = wrapper.forward_vl(image, text)

Hidden-state indexing follows HF: hidden_states[0] is the embedding output and
hidden_states[l] is the output of decoder layer l-1.
"""

import gc
import re
from typing import List, Optional, Tuple

import torch
from PIL import Image


_DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"


# ── Model name normalization ────────────────────────────────────────────────

def _normalize_model_name(model_id: str) -> str:
    """
    Normalize a HuggingFace model_id to a lowercase directory-safe name.

    Examples:
        llava-hf/llava-1.5-7b-hf           -> llava-1.5-7b-hf
        Qwen/Qwen2.5-VL-7B-Instruct        -> qwen2.5-vl-7b-instruct
    """
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"(\d)_(\d)", r"\1.\2", name)
    return name


def _default_bf16_dtype() -> torch.dtype:
    """bf16 where the GPU supports it natively, else fp16 (e.g. T4 / sm_75)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def expand2square(image: Image.Image, background_color) -> Image.Image:
    """Pad an image to a square canvas, centred (original LLaVA preprocessing)."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    result = Image.new(image.mode, (side, side), background_color)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


def _to_device(inputs, device) -> dict:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


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

    def prepare_vl(self, image: Image.Image, text: str) -> dict:
        raise NotImplementedError

    def prepare_text(self, text: str) -> dict:
        raise NotImplementedError

    def get_image_token_span(
        self, input_ids: torch.Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        raise NotImplementedError

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return self.processor.decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ── Shared forward / generation on prepared inputs ─────────────────────
    def _forward(self, inputs: dict, output_attentions: bool):
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

    def _generate(self, inputs: dict, max_new_tokens: int) -> str:
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        return self.decode(generated[0][input_len:]).strip()

    def forward_vl(self, image: Image.Image, text: str,
                   output_attentions: bool = False):
        return self._forward(self.prepare_vl(image, text), output_attentions)

    def forward_text(self, text: str, output_attentions: bool = False):
        return self._forward(self.prepare_text(text), output_attentions)

    def generate_vl(self, image: Image.Image, text: str,
                    max_new_tokens: int = 512) -> str:
        return self._generate(self.prepare_vl(image, text), max_new_tokens)

    def generate_text(self, text: str, max_new_tokens: int = 512) -> str:
        return self._generate(self.prepare_text(text), max_new_tokens)

    def generate_caption(self, image: Image.Image,
                         prompt: Optional[str] = None,
                         max_new_tokens: int = 512) -> str:
        return self.generate_vl(image, prompt or self._caption_prompt,
                                max_new_tokens=max_new_tokens)

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


# ── LLaVA-1.5 wrapper ───────────────────────────────────────────────────────

_LLAVA_CONFIGS = {
    "llava-hf/llava-1.5-7b-hf": {
        "num_image_tokens": 576,
        "prompt_template": "USER: <image>\n{text} ASSISTANT:",
        "text_only_template": "USER: {text} ASSISTANT:",
        "caption_prompt": "Please describe this image in detail.",
    },
}


class LLaVAWrapper(VLMWrapperBase):
    """Wrapper for LLaVA-1.5 (CLIP ViT-L/14@336 + LLaMA-2/Vicuna-7B, 32 layers).

    `pad_to_square=True` pads every image to a square with the CLIP mean colour
    before the HF processor runs, reproducing original LLaVA preprocessing
    instead of the HF processor's resize + centre crop.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL,
                 pad_to_square: bool = False, **kwargs):
        super().__init__(model_id, **kwargs)
        cfg = _LLAVA_CONFIGS.get(model_id, _LLAVA_CONFIGS[_DEFAULT_MODEL])
        self._num_image_tokens_cfg = cfg["num_image_tokens"]
        self._prompt_template = cfg["prompt_template"]
        self._text_only_template = cfg["text_only_template"]
        self._caption_prompt = cfg["caption_prompt"]
        self.pad_to_square = pad_to_square

    def load(self) -> "LLaVAWrapper":
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        print(f"Loading processor from '{self.model_id}'...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

        print(f"Loading model (LlavaForConditionalGeneration) from '{self.model_id}'...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        self._print_diagnostics()
        print(f"  num_image_tokens  : {self.num_image_tokens}")
        print(f"  image_token_id    : {self.image_token_id}")
        print(f"  pad_to_square     : {self.pad_to_square}")
        return self

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
    def get_image_token_span(
        self, input_ids: torch.Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        """[start, end) of the image tokens in the decoder sequence.

        Handles both processor conventions: a single `<image>` placeholder that
        the model expands to `num_image_tokens`, or ids already expanded by the
        processor (transformers >= 4.47).
        """
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        matches = (row == self.image_token_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            return None, None
        start = int(matches[0].item())
        if len(matches) == 1:
            return start, start + self.num_image_tokens
        return start, int(matches[-1].item()) + 1

    def get_text_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        img_start, img_end = self.get_image_token_span(input_ids)
        if img_start is None:
            return list(range(row.shape[0]))
        n_matches = int((row == self.image_token_id).sum().item())
        seq_len = row.shape[0] if n_matches > 1 else row.shape[0] - 1 + self.num_image_tokens
        return list(range(img_start)) + list(range(img_end, seq_len))

    # ── Input preparation ──────────────────────────────────────────────────
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.pad_to_square:
            mean = self.processor.image_processor.image_mean
            image = expand2square(image, tuple(int(x * 255) for x in mean))
        return image

    def prepare_vl(self, image: Image.Image, text: str) -> dict:
        prompt = self._prompt_template.format(text=text)
        inputs = self.processor(text=prompt, images=self.preprocess_image(image),
                                return_tensors="pt")
        return _to_device(inputs, self.device)

    def prepare_text(self, text: str) -> dict:
        prompt = self._text_only_template.format(text=text)
        inputs = self.processor(text=prompt, return_tensors="pt")
        return _to_device(inputs, self.device)


# ── Qwen2.5-VL wrapper ──────────────────────────────────────────────────────

_QWEN_VISION_START = "<|vision_start|>"
_QWEN_VISION_END = "<|vision_end|>"
_QWEN_IMAGE_PAD = "<|image_pad|>"

_QWEN_CONFIGS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "caption_prompt": "Please describe this image in detail.",
    },
}


class Qwen2VLWrapper(VLMWrapperBase):
    """
    Wrapper for Qwen/Qwen2.5-VL-7B-Instruct.

    LLM backbone: Qwen2.5-7B, 28 layers, hidden_dim=3584. No BOS token.
    Visual tokens use a dynamic resolution scheme; the processor expands
    the <|image_pad|> marker into the correct number of vision tokens, and
    rotary positions are multimodal (M-RoPE).

    Defaults to bf16 where supported. On GPUs without bf16 (T4) it falls back
    to fp16; check outputs for NaNs/garbage before trusting fp16 runs.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 max_pixels: Optional[int] = None, **kwargs):
        kwargs.setdefault("torch_dtype", _default_bf16_dtype())
        super().__init__(model_id, **kwargs)
        cfg = _QWEN_CONFIGS.get(model_id, _QWEN_CONFIGS["Qwen/Qwen2.5-VL-7B-Instruct"])
        self._caption_prompt = cfg["caption_prompt"]
        # None -> processor's native resolution. Pass an int to cap vision
        # tokens on constrained VRAM.
        self._max_pixels = max_pixels

    def load(self) -> "Qwen2VLWrapper":
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                f"Qwen2.5-VL requires transformers>=4.49. Original error: {e}"
            ) from e

        print(f"Loading processor from '{self.model_id}'...")
        proc_kwargs = {}
        if self._max_pixels is not None:
            proc_kwargs["max_pixels"] = self._max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_id, **proc_kwargs)
        self.tokenizer = self.processor.tokenizer
        # transformers>=4.49 drives smart_resize from size["longest_edge"], not
        # the max_pixels attribute, so a cap passed only via max_pixels is inert.
        if self._max_pixels is not None:
            ip = self.processor.image_processor
            ip.max_pixels = self._max_pixels
            size = dict(getattr(ip, "size", {}) or {})
            size["longest_edge"] = self._max_pixels
            size.setdefault("shortest_edge", getattr(ip, "min_pixels", 3136))
            ip.size = size

        print(f"Loading model (Qwen2_5_VLForConditionalGeneration) from '{self.model_id}'...")
        load_kwargs = dict(torch_dtype=self.torch_dtype, device_map=self.device_map)
        # Under 20 GiB VRAM the model (~16.6 GiB) needs CPU offload; reserve
        # headroom for vision-encoder intermediates and layer swaps.
        if self.device_map == "auto" and torch.cuda.is_available():
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gib < 20:
                gpu_budget = f"{int(total_vram_gib) - 5}GiB"
                load_kwargs["max_memory"] = {0: gpu_budget, "cpu": "24GiB"}
                print(f"  GPU < 20 GiB detected ({total_vram_gib:.1f} GiB) — "
                      f"capping GPU budget to {gpu_budget}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, **load_kwargs)
        self.model.eval()
        self._print_diagnostics()
        ip = self.processor.image_processor
        print(f"  max_pixels        : {getattr(ip, 'max_pixels', 'unknown')}")
        return self

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def num_layers(self) -> int:
        cfg = self.model.config
        if getattr(cfg, "text_config", None) is not None:
            return cfg.text_config.num_hidden_layers
        return cfg.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        cfg = self.model.config
        if getattr(cfg, "text_config", None) is not None:
            return cfg.text_config.hidden_size
        return cfg.hidden_size

    @property
    def num_attention_heads(self) -> int:
        cfg = self.model.config
        if getattr(cfg, "text_config", None) is not None:
            return cfg.text_config.num_attention_heads
        return cfg.num_attention_heads

    # ── Image/text token position helpers ──────────────────────────────────
    def _token_id(self, token: str) -> Optional[int]:
        tid = self.processor.tokenizer.convert_tokens_to_ids(token)
        if tid is None or tid == self.processor.tokenizer.unk_token_id:
            return None
        return tid

    def get_image_token_span(
        self, input_ids: torch.Tensor
    ) -> Tuple[Optional[int], Optional[int]]:
        """[start, end) of the visual tokens, exclusive of the vision markers."""
        vs_id = self._token_id(_QWEN_VISION_START)
        ve_id = self._token_id(_QWEN_VISION_END)
        if vs_id is None or ve_id is None:
            return None, None
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        starts = (row == vs_id).nonzero(as_tuple=True)[0]
        ends = (row == ve_id).nonzero(as_tuple=True)[0]
        if len(starts) == 0 or len(ends) == 0:
            return None, None
        return int(starts[0].item()) + 1, int(ends[0].item())

    def get_text_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        vision_ids = {self._token_id(t) for t in
                      (_QWEN_VISION_START, _QWEN_VISION_END, _QWEN_IMAGE_PAD)}
        vision_ids.discard(None)
        row = input_ids[0] if input_ids.dim() == 2 else input_ids
        return [i for i, tok in enumerate(row.tolist()) if tok not in vision_ids]

    # ── Input preparation ──────────────────────────────────────────────────
    def _build_prompt(self, text: str, image: Optional[Image.Image] = None) -> str:
        content = [{"type": "text", "text": text}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def prepare_vl(self, image: Image.Image, text: str) -> dict:
        image = image.convert("RGB")
        prompt = self._build_prompt(text, image=image)
        inputs = self.processor(text=[prompt], images=[image],
                                return_tensors="pt", padding=True)
        return _to_device(inputs, self.device)

    def prepare_text(self, text: str) -> dict:
        prompt = self._build_prompt(text)
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        return _to_device(inputs, self.device)


# ── Factory ─────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ("llava-hf/llava-1.5-7b-hf", "Qwen/Qwen2.5-VL-7B-Instruct")


def create_wrapper(model_id: str = _DEFAULT_MODEL, **kwargs) -> VLMWrapperBase:
    """Return the appropriate wrapper subclass for the given model_id.

    Wrapper-specific kwargs (`pad_to_square` for LLaVA, `max_pixels` for Qwen)
    are dropped silently for the other family so callers can pass both.
    """
    m = model_id.lower()
    if "llava" in m:
        kwargs.pop("max_pixels", None)
        return LLaVAWrapper(model_id, **kwargs)
    if "qwen2.5-vl" in m or "qwen2_5_vl" in m:
        kwargs.pop("pad_to_square", None)
        return Qwen2VLWrapper(model_id, **kwargs)
    raise ValueError(
        f"Unknown model: '{model_id}'. Supported: {', '.join(SUPPORTED_MODELS)}."
    )
