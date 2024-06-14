import os
import huggingface_hub


# Replace with the path to your.ckpt file
ckpt_file_path = "checkpoints2/hubert10.ckpt"
# Replace with the desired name for your repository
repo_name = "tarsssss/hubert-sslepoch_v1"


# Replace with your Hugging Face token
token = "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"

# Authenticate with the Hugging Face Hub
huggingface_hub.login(token)

# Create a new model repository
repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)

# Upload the.ckpt file
huggingface_hub.upload_file(
    path_or_fileobj=ckpt_file_path,
    path_in_repo="hubert10.ckpt",
    repo_id="tarsssss/hubert-sslepoch_v1",
    repo_type="model",
)
print("Model uploaded successfully!")