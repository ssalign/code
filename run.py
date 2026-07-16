import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

sys.path.insert(0, os.path.abspath("./src"))

from diffusers import FluxKontextPipeline  # noqa: E402
from diffusers.utils import load_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FLUX Kontext transfer with an optional appearance reference image. "
            "When --reference-image is provided, the script first inverts that image "
            "and feeds the inverted latent trajectory into the transfer pass."
        )
    )
    parser.add_argument("--model-path", default=os.getenv("FLUX_KONTEXT_MODEL", "black-forest-labs/FLUX.1-Kontext-dev"))
    parser.add_argument("--lora-path", default=os.getenv("FLUX_TRANSFER_LORA"))
    parser.add_argument("--input-image", required=True, help="Structure/content image to edit.")
    parser.add_argument("--chart-type", required=True, choices=["bar", "pie", "bubble", "line"], help="Input chart type.")
    parser.add_argument("--object", required=True, help="Semantic object used to transform chart elements.")
    parser.add_argument("--reference-image", default=None, help="Optional appearance reference image.")
    parser.add_argument("--prompt", default=None, help="Optional custom prompt for the edited output.")
    parser.add_argument(
        "--reference-prompt",
        default=None,
        help="Optional custom prompt used for reconstructing the reference image.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output root directory. Results are saved to {output_dir}/{chart}_{object}/{with_reference|no_reference}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--reference-guidance-scale", type=float, default=2.5)
    parser.add_argument("--appearance-align", default=True, action="store_true")
    parser.add_argument("--kv-alpha-max", type=float, default=0)
    parser.add_argument("--kv-start-ratio", type=float, default=0.2)
    parser.add_argument("--kv-end-ratio", type=float, default=0.8)
    parser.add_argument("--kv-schedule", choices=["step", "linear", "cosine"], default="step")
    parser.add_argument("--structure-align",default=True, action="store_true")
    parser.add_argument("--q-beta", type=float, default=1.0)
    parser.add_argument("--q-start", type=float, default=0.0)
    parser.add_argument("--q-end", type=float, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-auto-resize", action="store_true")
    return parser.parse_args()


PROMPT_TEMPLATES = {
    "bar": (
        "Transform the bar chart into a semantic image where each bar is replaced by a {object}. "
        "Each {object} must exactly match the original bar's height and proportion, preserving all relative heights, "
        "order, spacing, position and proportions. Realistic style, no text."
    ),
    "pie": (
        "Transform the pie chart into a semantic image where each slice is replaced by a segment of a {object}. "
        "Each segment of the {object} must exactly match the original slice's angle and proportion, preserving all "
        "relative angles, order, spacing, position and proportions. Realistic style, no text."
    ),
    "bubble": (
        "Transform the bubble chart into a semantic image where each point is replaced by a {object}. "
        "Each {object} must exactly match the original point's position and area, preserving all relative positions, "
        "spacing, and proportions. Realistic style, no text."
    ),
    "line": (
        "Transform the line chart into a semantic image where each segment is replaced by a {object}. "
        "Each {object} must exactly match the original segment's position, preserving all relative positions, order, "
        "spacing, and proportions. Realistic style, no text."
    ),
}

REFERENCE_PROMPT_TEMPLATE = (
    "A realistic image of a {object}, front-facing view, isolated on a plain white background, no additional objects."
)


def resolve_prompts(args: argparse.Namespace) -> None:
    if args.prompt is None:
        args.prompt = PROMPT_TEMPLATES[args.chart_type].format(object=args.object)
    if args.reference_prompt is None:
        args.reference_prompt = REFERENCE_PROMPT_TEMPLATE.format(object=args.object)


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_pipeline(args: argparse.Namespace) -> FluxKontextPipeline:
    pipe = FluxKontextPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype(args.dtype))
    pipe.to(args.device)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    return pipe


def invert_reference(
    pipe: FluxKontextPipeline,
    reference_image_path: str,
    reference_prompt: str,
    args: argparse.Namespace,
) -> List[torch.Tensor]:
    reference_image = Image.open(reference_image_path).convert("RGB")
    reference_latents = pipe.encode_image_to_packed_latents(
        reference_image,
        height=args.height,
        width=args.width,
        device=args.device,
    )
    inverted_latents = pipe(
        prompt=reference_prompt,
        height=args.height,
        width=args.width,
        guidance_scale=1.0,
        output_type="latent",
        num_inference_steps=args.steps,
        max_sequence_length=512,
        latents=reference_latents,
        invert_image=True,
        _auto_resize=not args.disable_auto_resize,
    )
    return [latent.detach().to("cpu", dtype=torch.float16).contiguous() for latent in inverted_latents]


def build_batch(args: argparse.Namespace, has_reference: bool):
    input_image = load_image(args.input_image).convert("RGB")
    if not has_reference:
        return args.prompt, input_image, args.guidance_scale

    blank_reference_context = Image.new("RGB", (args.width, args.height), 0)
    prompts = [args.reference_prompt, args.prompt]
    images = [blank_reference_context, input_image]
    guidance = [args.reference_guidance_scale, args.guidance_scale]
    return prompts, images, guidance


def sanitize_path_part(value: str, fallback: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or fallback


def resolve_output_dir(args: argparse.Namespace, has_reference: bool) -> Path:
    chart_name = sanitize_path_part(args.chart_type, "chart")
    object_name = sanitize_path_part(args.object, "object")
    mode_name = "with_reference" if has_reference else "no_reference"
    return Path(args.output_dir) / f"{chart_name}_{object_name}" / mode_name


def save_outputs(images, args: argparse.Namespace, seed: int, has_reference: bool) -> None:
    output_dir = resolve_output_dir(args, has_reference)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(images, list):
        images = [images]

    start_index = 1 if has_reference and len(images) > 1 else 0
    for idx, image in enumerate(images[start_index:], start=0):
        image.save(output_dir / f"transfer_seed_{seed}_{idx:02d}.png")

    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        if has_reference:
            f.write(f"reference_prompt: {args.reference_prompt}\n")
            f.write(f"reference_image: {args.reference_image}\n")
        f.write(f"chart_type: {args.chart_type}\n")
        f.write(f"object: {args.object}\n")
        f.write(f"prompt: {args.prompt}\n")
        f.write(f"input_image: {args.input_image}\n")
        f.write(f"output_dir: {output_dir}\n")


def main() -> None:
    args = parse_args()
    resolve_prompts(args)
    pipe = load_pipeline(args)
    has_reference = args.reference_image is not None

    inverted_latent_list: Optional[List[List[torch.Tensor]]] = None
    if has_reference:
        inverted_latent_list = [invert_reference(pipe, args.reference_image, args.reference_prompt, args)]

    prompt, image, guidance_scale = build_batch(args, has_reference)

    for offset in range(args.num_images):
        seed = args.seed + offset
        generator = torch.Generator(device=args.device).manual_seed(seed)
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=generator,
            kv_interpolate=args.appearance_align,
            inverted_latent_list=inverted_latent_list,
            kv_alpha_max=args.kv_alpha_max,
            kv_start_ratio=args.kv_start_ratio,
            kv_end_ratio=args.kv_end_ratio,
            kv_schedule=args.kv_schedule,
            app_object=args.object,
            structure_align=args.structure_align,
            q_beta=args.q_beta,
            q_start=args.q_start,
            q_end=args.q_end,
            _auto_resize=not args.disable_auto_resize,
        )
        save_outputs(result.images, args, seed, has_reference)


if __name__ == "__main__":
    main()
