from .nodes.s3_save_image import S3SaveImage

NODE_CLASS_MAPPINGS = {
    "Save Image To S3": S3SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Image To S3": "💾 Save Your Image to S3",
}
