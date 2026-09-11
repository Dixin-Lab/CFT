import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import os
import torchvision.transforms.functional as F

Image.MAX_IMAGE_PIXELS = None

def make_train_dataset(args, accelerator=None, data_group=None):
    if args.train_data_dir is not None:
        print("loading dataset ... ")
        dataset = load_dataset('json', data_files=args.train_data_dir)
        base_path = os.path.dirname(os.path.abspath(args.train_data_dir))

    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.caption_column is None:
        caption_column = column_names[0]
        print(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.source_column is None:
        source_column = column_names[1]
        print(f"source column defaulting to {source_column}")
    else:
        source_column = args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"`--source_column` value '{args.source_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
    if args.target_column is None:
        target_column = column_names[2]
        print(f"target column defaulting to {target_column}")
    else:
        target_column = args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"`--target_column` value '{args.target_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def resize_long_side(img, target_long_side, interpolation=transforms.InterpolationMode.BILINEAR):
        w, h = img.size
        if w >= h:
            new_w = target_long_side
            new_h = int(target_long_side * h / w)
        else:
            new_h = target_long_side
            new_w = int(target_long_side * w / h)
        return F.resize(img, (new_h, new_w), interpolation=interpolation)

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    trains = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    def preprocess_train(examples):
        _examples = {}

        source_images = []
        target_images = []
        ref_src_images = []
        ref_tgt_images = []
        captions = []
        for source_image_path, target_image_path, caption in zip(examples[source_column], examples[target_column], examples[caption_column]):
            source_image = Image.open(os.path.join(args.dataset_root_path, source_image_path)).convert("RGB")
            target_image = Image.open(os.path.join(args.dataset_root_path, target_image_path)).convert("RGB")
            source_images.append(source_image)
            target_images.append(target_image)
            captions.append(caption)

            group_id = caption.strip()

            exemplar_candidates = data_group[group_id]
            exemplar_item = random.choice(exemplar_candidates)
            while exemplar_item['source'] == source_image_path:
                exemplar_item = random.choice(exemplar_candidates)
            exemplar_source_image_path = os.path.join(args.dataset_root_path, exemplar_item['source'])
            exemplar_target_image_path = os.path.join(args.dataset_root_path, exemplar_item['target'])
            exemplar_source_image = Image.open(exemplar_source_image_path).convert('RGB')
            exemplar_target_image = Image.open(exemplar_target_image_path).convert('RGB')
            ref_src_images.append(exemplar_source_image)
            ref_tgt_images.append(exemplar_target_image)

        _examples["cond_pixel_values"] = [train_transforms(source) for source in source_images]
        _examples["pixel_values"] = [train_transforms(image) for image in target_images]
        _examples["qwen_cond_images"] = [trains(image) for image in source_images]
        _examples["captions"] = examples[caption_column]
        _examples["ref_src_pixel_values"] = [train_transforms(image) for image in ref_src_images]
        _examples["ref_tgt_pixel_values"] = [train_transforms(image) for image in ref_tgt_images]

        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset

def collate_fn(examples):
    cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    ref_src_pixel_values = torch.stack([example["ref_src_pixel_values"] for example in examples])
    ref_src_pixel_values = ref_src_pixel_values.to(memory_format=torch.contiguous_format).float()
    ref_tgt_pixel_values = torch.stack([example["ref_tgt_pixel_values"] for example in examples])
    ref_tgt_pixel_values = ref_tgt_pixel_values.to(memory_format=torch.contiguous_format).float()

    qwen_cond_images = [example["qwen_cond_images"] for example in examples]
    captions = [example["captions"] for example in examples]

    return {
        "cond_pixel_values": cond_pixel_values,
        "qwen_cond_images": qwen_cond_images,
        "pixel_values": target_pixel_values,
        "ref_src_pixel_values": ref_src_pixel_values,
        "ref_tgt_pixel_values": ref_tgt_pixel_values,
        "captions": captions,

    }
