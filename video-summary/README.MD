# Video Summary Generator

A Python tool that generates frame-by-frame and overall summaries of video content using AI vision models.

## Features

- Processes video files (supports MP4, WebM, and other common formats)
- Generates detailed descriptions for individual frames
- Creates an overall summary of the entire video
- Supports both OpenAI and Anthropic APIs for vision analysis
- Configurable frame sampling rate

## Prerequisites

- Python 3.10 or higher
- OpenCV (cv2)
- OpenAI API key or Anthropic API key

## Installation

1. Clone the repository:
git clone [your-repo-url]
cd video-summary

2. Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate # On Windows, use .venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

4. Create a `.env` file in the project root:
OPENAI_API_KEY=your-openai-key-here
or
ANTHROPIC_API_KEY=your-anthropic-key-here


## Usage

Run the script and follow the prompts:
python video_summary.py


When prompted, enter the path to your video file. The script will:
1. Process the video frame by frame
2. Generate descriptions for each frame
3. Create an overall summary
4. Save the results to a text file

## Configuration

You can adjust the following parameters in the code:
- `sample_rate`: Number of frames to skip (default: process 1 frame per second)
- `api_type`: Choose between "openai" or "anthropic" for vision analysis
- `max_tokens`: Adjust the length of generated descriptions

## Output

The script generates a text file containing:
- Individual frame summaries
- An overall video summary
- Timestamps for each analyzed frame

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

[Your chosen license]