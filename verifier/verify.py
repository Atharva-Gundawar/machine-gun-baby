# pip install google-genai wandb weave
from google import genai
from google.genai import types
import mimetypes
import os
from pydantic import BaseModel
import weave

# ---------- Config ----------
GEMINI_MODEL = "gemini-2.5-flash"   # or "gemini-2.5-pro" for heavier reasoning

# GEMINI_API_KEY must be set in your environment:
# export GEMINI_API_KEY=...

weave.init("quickstart")

class VerificationResult(BaseModel):
    """Structured output for condition verification"""
    condition_met: bool

def _infer_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

@weave.op()
def verify_video_condition(video_path: str, condition: str) -> VerificationResult:
    """
    Verify if a condition is met in a video.

    Args:
        video_path: Path to the video file
        condition: The condition to verify (e.g., "the video has a right-click in it")

    Returns:
        VerificationResult with condition_met (bool), confidence, and reasoning
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing file: {video_path}")

    mime = _infer_mime(video_path)
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    prompt = f"""You are a video verification system. Your task is to determine if the following condition was achieved in the video.

CONDITION: {condition}

Analyze the video carefully and determine if the condition was successfully achieved (true/false).

Be strict in your evaluation - the condition must be fully met to return true."""

    client = genai.Client()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=types.Content(parts=[
            types.Part(inline_data=types.Blob(data=video_bytes, mime_type=mime)),
            types.Part(text=prompt),
        ]),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationResult,
        )
    )

    result = VerificationResult.model_validate_json(resp.text)
    return result

if __name__ == "__main__":
    video_path = "rec.mp4"
    condition = "the video has a right-click in it"

    result = verify_video_condition(video_path, condition)

    print(f"Condition Met: {result.condition_met}")
