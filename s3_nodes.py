import os
from io import BytesIO
from typing import List

import boto3
import numpy as np
from easy_nodes import Choice, ComfyNode, ImageTensor, StringInput
from PIL import Image


@ComfyNode(
    category="image",
    display_name="Save Image to S3",
    description="Saves an image to an S3 bucket.",
    is_output_node=False,
)
def save_image_to_s3(
    image: ImageTensor,
    key: str = StringInput("folder/image.png"),
    bucket_name: str = StringInput("bucket_name"),
    access_key_id_env: str = StringInput("AWS_ACCESS_KEY_ID"),
    secret_access_key_env: str = StringInput("AWS_SECRET_ACCESS_KEY"),
    region_env: str = StringInput("AWS_REGION"),
    endpoint_url_env: str = StringInput("AWS_ENDPOINT_URL"),
    format: str = Choice(["PNG", "JPG", "JPEG", "WEBP"]),
):
    """
    Saves the provided image to the specified S3 bucket.
    """
    aws_access_key_id = os.getenv(access_key_id_env)
    aws_secret_access_key = os.getenv(secret_access_key_env)
    aws_region = os.getenv(region_env, "auto")
    aws_endpoint_url = os.getenv(endpoint_url_env, "https://s3.amazonaws.com")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials are not properly set in the environment variables."
        )

    s3_client = boto3.client(
        "s3",
        region_name=aws_region,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Convert tensor to PIL Image
    i = 255.0 * image.cpu().numpy()
    i = np.squeeze(i)
    if len(i.shape) == 4:
        i = i[0]  # Take first image if we have a batch
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # Convert PIL Image to bytes
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)

    s3_client.put_object(
        Body=img_byte_arr,
        Key=key,
        Bucket=bucket_name,
        ContentType=f"image/{format.lower()}",
    )

    return {"ui": {"images": [{"filename": key, "subfolder": "", "type": "output"}]}}
