import cv2
import os
import time
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
import io
import base64
import numpy as np
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class VideoSummarizer:
    def __init__(self, api_key, api_type="anthropic"):
        """
        Initialize the video summarizer with either Anthropic or OpenAI API

        Args:
            api_key (str): API key for the chosen service
            api_type (str): Either "anthropic" or "openai"
        """
        self.api_type = api_type
        if api_type == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)

    def extract_frames(self, video_path, sample_rate=1):
        """
        Extract frames from video at given sample rate

        Args:
            video_path (str): Path to input video file
            sample_rate (int): Extract every nth frame

        Returns:
            list: List of extracted frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

    def encode_frame(self, frame):
        """
        Encode frame as base64 string

        Args:
            frame (numpy.ndarray): Input frame

        Returns:
            str: Base64 encoded frame
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Convert to base64
        buff = io.BytesIO()
        pil_img.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode('utf-8')

    def get_frame_summary(self, frame):
        """
        Get summary of a single frame using the configured API
        """
        encoded_frame = self.encode_frame(frame)

        cursor_prompt = """
        Please describe what's happening in this frame from a screen recording, focusing specifically on:
        1. The location and movement of the cursor (shown as a red line/trail)
        2. Where the cursor came from and where it's going
        3. Any interactions or clicks the cursor is making
        4. The context of what's being interacted with on screen

        Keep the description brief but precise about cursor position and movement.
        """

        if self.api_type == "anthropic":
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": cursor_prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": encoded_frame
                            }
                        }
                    ]
                }]
            )
            return response.content[0].text

        else:  # OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4-vision",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": cursor_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_frame}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            return response.choices[0].message.content

    def process_frame(self, frame, frame_number):
        """
        Process a single frame and return its summary with frame number
        """
        try:
            summary = self.get_frame_summary(frame)
            print(f"Processed frame {frame_number}")
            return frame_number, summary
        except Exception as e:
            print(f"Error processing frame {frame_number}: {str(e)}")
            return frame_number, f"Error: {str(e)}"

    def summarize_video(self, video_path, sample_rate=30, max_workers=4):
        """
        Generate a complete summary of the video using parallel processing

        Args:
            video_path (str): Path to input video file
            sample_rate (int): Sample every nth frame
            max_workers (int): Maximum number of concurrent threads
        """
        frames = self.extract_frames(video_path, sample_rate)
        frame_summaries = [None] * len(frames)  # Pre-allocate list
        print(f"Processing {len(frames)} frames using {max_workers} threads...")

        # Create a thread pool and process frames concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frames for processing
            future_to_frame = {
                executor.submit(self.process_frame, frame, i): i
                for i, frame in enumerate(frames)
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_num, summary = future.result()
                frame_summaries[frame_num] = summary

        # Remove any None values in frame_summaries
        frame_summaries = [summary for summary in frame_summaries if summary is not None]

        # Generate overall summary
        combined_summary = "\n".join(frame_summaries)
        prompt = f"Based on these frame descriptions, provide a coherent summary of the video:\n{combined_summary}"

        if self.api_type == "anthropic":
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            overall_summary = response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            overall_summary = response.choices[0].message.content

        return frame_summaries, overall_summary

# Example usage
def main():
    # Load environment variables
    load_dotenv()

    # Get API key from environment variables
    api_type = "anthropic"  # or "anthropic"

    if api_type == "anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize summarizer with the correct API type
    summarizer = VideoSummarizer(api_key, api_type=api_type)

    # Get video path from user and clean it
    video_path = input("Please enter the path to your video file: ").strip()
    video_path = os.path.normpath(video_path)  # Normalize path (fix double slashes)

    # Verify file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Verify we can open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}. Please ensure the file is a valid video format.")
    cap.release()

    # Process video
    frame_summaries, overall_summary = summarizer.summarize_video(
        video_path,
        sample_rate=30,  # Process 1 frame every second for 30fps video
        max_workers=4
    )

    # Save results
    with open("video_summary.txt", "w") as f:
        f.write("Frame Summaries:\n")
        for i, summary in enumerate(frame_summaries):
            f.write(f"\nFrame {i+1}:\n{summary}\n")

        f.write("\nOverall Video Summary:\n")
        f.write(overall_summary)

if __name__ == "__main__":
    main()