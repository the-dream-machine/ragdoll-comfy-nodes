import os
from io import BytesIO

import boto3
from easy_nodes import ComfyNode, ImageTensor, StringInput, show_image
from PIL import Image


@ComfyNode(
    category="image",
    display_name="Save Image to S3",
    description="Saves an image to an S3 bucket and returns the image for preview.",
)
def save_image_to_s3(
    image: ImageTensor,
    key: str = StringInput("folder/image.png"),  # Supports full path
    bucket_name: str = StringInput("YourBucketName"),
    access_key_id_env: str = StringInput("AWS_ACCESS_KEY_ID"),
    secret_access_key_env: str = StringInput("AWS_SECRET_ACCESS_KEY"),
    region_env: str = StringInput("AWS_REGION"),  # Default region
    endpoint_url_env: str = StringInput("AWS_ENDPOINT_URL"),  # Default endpoint
) -> ImageTensor:
    """
    Saves the provided image to the specified S3 bucket and returns the image for preview.
    """
    # Retrieve AWS credentials and configuration from environment variables
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

    # Convert the image tensor to a PIL Image
    pil_image = Image.fromarray((image * 255).astype("uint8"))

    # Save the image to a BytesIO object
    image_buffer = BytesIO()
    pil_image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    # Upload the image to S3
    s3_client.upload_fileobj(image_buffer, bucket_name, key)

    # Display the image as a preview in the node
    show_image(pil_image)

    # Return the image for further processing or preview
    return image
