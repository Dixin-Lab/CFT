import argparse
import copy
import logging
import math
import os
import sys
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
import datetime

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from typing import List, Optional, Union

from PIL import Image
import numpy as np
import torch.utils.checkpoint
import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

import diffusers
from diffusers.utils import check_min_version, is_wandb_available

from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")
logger = get_logger(__name__)

def log_validation(
        pipeline,
        args,
        accelerator,
        pipeline_args,
        step,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images

def get_transformer_layer_cls():
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionBlock, Qwen2_5_VLDecoderLayer
    return {
        QwenImageTransformerBlock,

        Qwen2_5_VLVisionBlock,
        Qwen2_5_VLDecoderLayer
        }

def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="CFT LoRA training for Qwen-Image-Edit-2509.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Path to the training JSONL containing source, target, and text fields.",
    )
    parser.add_argument(
        "--json_for_ref",
        type=str,
        default=None,
        help="Path to the JSON mapping each stripped text condition to reference image pairs.",
    )
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        default=None,
        help="Root directory for image paths in the training data and reference groups.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="control_left",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--ipa1_column",
        type=str,
        default="control_left",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--ipa2_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--ipa3_column",
        type=str,
        default="control_left",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption_left",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="Change the cat's fur to white",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every X epochs. validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./canny_model_512",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="qwen-image-lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="The alpha parameter for the consistency loss.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result

def _get_qwen_prompt_embeds(
    text_encoder,
    processor,
    prompt: Union[str, List[str]] = None,
    image: Optional[Image.Image] = None,
    device: Optional[torch.device] = None,
    weight_dtype = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    if isinstance(image, list):
        base_img_prompt = ""
        for i, img in enumerate(image):
            base_img_prompt += img_prompt_template.format(i + 1)
    elif image is not None:
        base_img_prompt = img_prompt_template.format(1)
    else:
        base_img_prompt = ""

    template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    drop_idx = 64
    txt = [template.format(base_img_prompt + e) for e in prompt]

    model_inputs = processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = _extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)

    return prompt_embeds, encoder_attention_mask

def encode_prompt(
    text_encoder,
    qwen_processor,
    prompt: Union[str, List[str]],
    image: Optional[Image.Image] = None,
    device: Optional[torch.device] = None,
    dtype=None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 4064
):
    r"""
    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        image (`torch.Tensor`, *optional*):
            image to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    """
    device = device or text_encoder.device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(text_encoder, qwen_processor, prompt, image, device, dtype)

    prompt_embeds = prompt_embeds[:, :max_sequence_length]
    prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask

def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def _encode_vae_image(vae, image: torch.Tensor, generator: torch.Generator = None):
    latent_channels = vae.config.z_dim if vae is not None else 16
    image_latents = retrieve_latents(vae.encode(image), generator=generator, sample_mode="argmax")
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std)
        .view(1, latent_channels, 1, 1, 1)
        .to(image_latents.device, image_latents.dtype)
    )
    image_latents = (image_latents - latents_mean) / latents_std
    return image_latents

def main(args):
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        project_dir=os.path.join(args.output_dir, "wandb_logs"),
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        # 保存训练参数和系统信息
        script_path = os.path.abspath(__file__)
        print(f"脚本完整路径: {script_path}")
        # 收集训练参数和系统信息
        training_info = {
            "script_path": script_path,
            "training_args": vars(args),
            "system_info": {
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
                "python_version": sys.version,
                "diffusers_version": diffusers.__version__,
                "accelerate_version": accelerate.__version__,
                "transformers_version": transformers.__version__,
            },
            "timestamp": str(datetime.datetime.now())
        }

        # 保存为 JSON 格式
        info_path = os.path.join(args.output_dir, "training_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        # 保存为可读的文本格式
        readable_path = os.path.join(args.output_dir, "training_info.txt")
        with open(readable_path, "w", encoding="utf-8") as f:
            f.write("Training Arguments:\n")
            f.write("=" * 50 + "\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
            f.write("\nSystem Information:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}\n")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
                f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"Python Version: {sys.version}\n")
            f.write(f"Diffusers Version: {diffusers.__version__}\n")
            f.write(f"Accelerate Version: {accelerate.__version__}\n")
            f.write(f"Transformers Version: {transformers.__version__}\n")
            f.write(f"Timestamp: {datetime.datetime.now()}\n")

    # Load the tokenizers
    vae = AutoencoderKLQwenImage.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = QwenImageTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    logger.info("All models loaded successfully")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    processor = Qwen2VLProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="processor")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "img_in",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    if args.train_norm_layers:
                        transformer_norm_layers_to_save = {
                            f"transformer.{name}": param
                            for name, param in model.named_parameters()
                            if any(k in name for k in NORM_LAYER_PREFIXES)
                        }
                        transformer_lora_layers_to_save = {
                            **transformer_lora_layers_to_save,
                            **transformer_norm_layers_to_save,
                        }
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            QwenImageEditPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(flux_transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = QwenImageTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            ).to(accelerator.device, weight_dtype)

            # Handle input dimension doubling before adding adapter
            with torch.no_grad():
                initial_input_channels = transformer_.config.in_channels
                new_linear = torch.nn.Linear(
                    transformer_.x_embedder.in_features * 2,
                    transformer_.x_embedder.out_features,
                    bias=transformer_.x_embedder.bias is not None,
                    dtype=transformer_.dtype,
                    device=transformer_.device,
                )
                new_linear.weight.zero_()
                new_linear.weight[:, :initial_input_channels].copy_(transformer_.x_embedder.weight)
                if transformer_.x_embedder.bias is not None:
                    new_linear.bias.copy_(transformer_.x_embedder.bias)
                transformer_.x_embedder = new_linear
                transformer_.register_to_config(in_channels=initial_input_channels * 2)

            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = QwenImageEditPipeline.lora_state_dict(input_dir)
        transformer_lora_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        }
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_lora_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_norm_layers:
            transformer_norm_state_dict = {
                k: v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            transformer_._transformer_norm_layers = QwenImageEditPipeline._load_norm_into_transformer(
                transformer_norm_state_dict,
                transformer=transformer_,
                discard_original_layers=False,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        global_step = int(path.split("-")[-1])
        initial_global_step = global_step
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator._models.append(transformer)
        accelerator.load_state(path)
        first_epoch = 0
    else:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    from dataset import make_train_dataset, collate_fn
    with open(args.json_for_ref, 'r') as infile:
        data_group = json.load(infile)
    # Dataset and DataLoaders creation:
    train_dataset = make_train_dataset(args, accelerator, data_group)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):

                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)
                ref_src_pixel_values = batch["ref_src_pixel_values"].to(dtype=vae.dtype)
                ref_tgt_pixel_values = batch["ref_tgt_pixel_values"].to(dtype=vae.dtype)

                model_input = _encode_vae_image(vae, pixel_values.unsqueeze(2))  #注意这里vae的编码结果有5个维度，比正常flux的多一个
                model_input = model_input.to(dtype=weight_dtype)

                cond_input = _encode_vae_image(vae, cond_pixel_values.unsqueeze(2))
                cond_input = cond_input.to(dtype=weight_dtype)

                ref_src_input = _encode_vae_image(vae, ref_src_pixel_values.unsqueeze(2))
                ref_src_input = ref_src_input.to(dtype=weight_dtype)

                ref_tgt_input = _encode_vae_image(vae, ref_tgt_pixel_values.unsqueeze(2))
                ref_tgt_input = ref_tgt_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** len(vae.temperal_downsample) if vae is not None else 8

                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = encode_prompt(
                        text_encoder,
                        processor,
                        prompt=batch["captions"]+["Keep the image unchanged."]*len(batch["captions"]),
                        image=[batch["qwen_cond_images"]*2],
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0] * 2

                # Sample a random timestep for each image

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz//2,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input_1 = (1.0 - sigmas) * model_input + sigmas * noise
                noisy_model_input_2 = (1.0 - sigmas) * cond_input + sigmas * noise

                noisy_model_input_cat = torch.cat([noisy_model_input_1, noisy_model_input_2], dim=0)
                cond_input_cat = torch.cat([cond_input, cond_input], dim=0)
                timesteps = torch.cat([timesteps, timesteps], dim=0)

                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input_cat,
                    batch_size=bsz,
                    num_channels_latents=noisy_model_input_cat.shape[1],
                    height=noisy_model_input_cat.shape[3],
                    width=noisy_model_input_cat.shape[4],
                )

                packed_cond_model_input = QwenImageEditPipeline._pack_latents(
                    cond_input_cat,
                    batch_size=bsz,
                    num_channels_latents=cond_input_cat.shape[1],
                    height=cond_input_cat.shape[3],
                    width=cond_input_cat.shape[4],
                )

                img_shapes = [
                    [
                        (1, noisy_model_input_cat.shape[3]//2, noisy_model_input_cat.shape[4]//2),
                        (1, cond_input_cat.shape[3]//2, cond_input_cat.shape[4]//2),
                    ]
                ] * bsz

                concatenated_noisy_model_input = torch.concat([packed_noisy_model_input, packed_cond_model_input], dim=1)

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=concatenated_noisy_model_input,
                    timestep=timesteps / 1000,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    txt_seq_lens=txt_seq_lens,
                    img_shapes=img_shapes,
                    guidance=None,
                    return_dict=False,
                )[0]

                model_pred = model_pred[:, :packed_noisy_model_input.shape[1], :]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input_cat.shape[3]* vae_scale_factor,
                    width=noisy_model_input_cat.shape[4]* vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target_1 = noise - model_input
                target_2 = noise - cond_input

                target_3 = ref_src_input - ref_tgt_input
                model_pred_1, model_pred_2 = model_pred.chunk(2, dim=0)
                model_pred_3 = model_pred_1 - model_pred_2
                # Compute regular loss.
                loss_1 = torch.mean(
                    (weighting.float() * (model_pred_1.float() - target_1.float()) ** 2).reshape(target_1.shape[0], -1),
                    1,
                ).mean()
                loss_2 = torch.mean(
                    (weighting.float() * (model_pred_2.float() - target_2.float()) ** 2).reshape(target_2.shape[0], -1),
                    1,
                ).mean()
                loss_3 = torch.mean(
                    (weighting.float() * (model_pred_3.float() - target_3.float()) ** 2).reshape(target_3.shape[0], -1),
                    1,
                ).mean()

                loss = loss_1 + loss_2 + args.alpha * loss_3

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        transformer_lora_layers = get_peft_model_state_dict(unwrap_model(transformer))
                        QwenImageEditPipeline.save_lora_weights(
                            save_directory=save_path,
                            transformer_lora_layers=transformer_lora_layers,
                        )

            logs = {"loss": loss.detach().item(), "loss_1": loss_1.mean().detach().item(), "loss_2": loss_2.mean().detach().item(), "loss_3": loss_3.mean().detach().item(), "lr": lr_scheduler.get_last_lr()[0],"global_step": global_step, "epoch": epoch}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
