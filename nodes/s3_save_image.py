import io
import json
import os

import boto3
import numpy as np
from cuid2 import Cuid
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class S3SaveImage:
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
                        "default": "images.development.ragdollai.xyz",
                        "tooltip": "The name of the S3 bucket to save the images to.",
                    },
                ),
                "folder": (
                    "STRING",
                    {"default": "", "tooltip": "The folder to save the images to."},
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
        images,
        bucket="images.development.ragdollai.xyz",
        folder="",
        prompt=None,
        extra_pnginfo=None,
    ):
        results = []

        for image in images:
            # Convert tensor to image
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()

            # Add metadata
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))

            # Generate unique filename using cuid2
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
