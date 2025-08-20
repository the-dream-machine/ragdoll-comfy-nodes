import io
import json
import os
from typing import Any, Optional

import boto3
import numpy as np
import torch
from cuid2 import Cuid
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class S3SaveImages:
    def __init__(self):
        self.cuid: Cuid = Cuid()
        self.client = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", "auto"),
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "https://s3.amazonaws.com"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "bucket": (
                    "STRING",
                    {
                        "default": "my_image_bucket",
                        "tooltip": "The name of the S3 bucket to save the images to.",
                    },
                ),
                "folder": (
                    "STRING",
                    {"default": "", "tooltip": "The folder to save the images to."},
                ),
                "image_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Custom ID to use for the image filename. If empty, a unique ID will be generated.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "s3_save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to an S3 bucket."

    def s3_save_images(
        self,
        images: torch.Tensor,
        bucket: str = "my_image_bucket",
        folder: str = "",
        image_id: str = "",
        prompt: Optional[dict[str, Any]] = None,
        extra_pnginfo: Optional[dict[str, Any]] = None,
    ) -> "dict[str, dict[str, list[dict[str, str]]]]":
        results = []

        for index, image in enumerate(images):
            # Convert tensor to image
            img_array = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
            metadata = PngInfo()

            # Add metadata
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))

            # Generate filename
            if image_id:
                if len(images) > 1:
                    filename = f"{image_id}_{index}.png"
                else:
                    filename = f"{image_id}.png"
            else:
                unique_id = self.cuid.generate()
                filename = f"{unique_id}.png"

            s3_key = f"{folder}/{filename}" if folder else filename

            # Save image to in-memory file
            image_buffer = io.BytesIO()
            img.save(image_buffer, format="PNG", pnginfo=metadata)
            image_buffer.seek(0)  # Reset buffer position

            # Upload to S3
            self.client.upload_fileobj(image_buffer, bucket, s3_key)

            # Store result
            results.append({"filename": filename})

        return {"ui": {"images": results}}
