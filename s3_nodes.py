import os
from io import BytesIO
from typing import List
import numpy as np

import boto3
from easy_nodes import ComfyNode, ImageTensor, StringInput, show_image
from PIL import Image


@ComfyNode(
    category="image",
    display_name="Save Image to S3",
    description="Saves images to an S3 bucket and shows a preview of the images.",
    is_output_node=True,
)
def save_image_to_s3(
    images: List[ImageTensor],
    key: str = StringInput("folder/image_%batch_num%.png"),
    bucket_name: str = StringInput("YourBucketName"),
    access_key_id_env: str = StringInput("AWS_ACCESS_KEY_ID"),
    secret_access_key_env: str = StringInput("AWS_SECRET_ACCESS_KEY"),
    region_env: str = StringInput("AWS_REGION"),
    endpoint_url_env: str = StringInput("AWS_ENDPOINT_URL"),
)-> List[ImageTensor]:
    """
    Saves the provided images to the specified S3 bucket and shows a preview of the images.
    """
    aws_access_key_id = os.getenv(access_key_id_env)
    aws_secret_access_key = os.getenv(secret_access_key_env)
    aws_region = os.getenv(region_env, "auto")
    aws_endpoint_url = os.getenv(endpoint_url_env, "https://s3.amazonaws.com")

    # Check if essential credentials are available
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials are not properly set in the environment variables."
        )

    # Initialize the S3 client with the retrieved credentials and configuration
    s3_client = boto3.client(
        "s3",
        region_name=aws_region,
        endpoint_url=aws_endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    results = []
    for batch_number, image in enumerate(images):
        print(f"Image tensor: {image}")
        
        # Convert tensor to PIL Image
        i = np.squeeze(image.cpu().numpy()) * 255.
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        

        show_image(img)
        
        # Generate unique key for each image
        batch_key = key.replace("%batch_num%", str(batch_number))
        
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Upload to S3
        s3_client.upload_fileobj(
            img_byte_arr,
            bucket_name,
            batch_key
        )
        
        results.append(img)
    
    return results

