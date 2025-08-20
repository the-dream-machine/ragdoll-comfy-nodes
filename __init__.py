from .nodes.s3_save_images import S3SaveImages

NODE_CLASS_MAPPINGS = {
    "Save Images To S3": S3SaveImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Save Images To S3": "💾 Save Images to S3",
}
