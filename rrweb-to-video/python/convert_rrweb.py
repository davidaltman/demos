import json
import os
import tempfile
from pathlib import Path
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import time
from typing import List, Dict
import base64
import ffmpeg

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RRWebConverter:
    def __init__(self):
        self.driver = None
        self.temp_html = None
        self.fps = 30

    def create_html_player(self, events: List[Dict]) -> str:
        """Create HTML content for the rrweb player."""
        return f"""
        <!DOCTYPE html>
        <html>
          <head>
            <title>RRWeb Playback</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/style.css"/>
            <script src="https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/index.js"></script>
            <style>
              body {{ margin: 0; }}
              #player {{ width: 1920px; height: 1080px; }}
            </style>
          </head>
          <body>
            <div id="player"></div>
            <script>
              window.addEventListener('load', () => {{
                const events = {json.dumps(events)};
                const replayer = new rrwebPlayer({{
                  target: document.getElementById('player'),
                  data: {{
                    events,
                    autoPlay: true,
                  }}
                }});
                window.playerReady = true;
              }});
            </script>
          </body>
        </html>
        """

    def setup_driver(self):
        """Initialize the Chrome driver with appropriate options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)

    def record_browser(self, duration: float, output_path: str):
        """Record the browser content for the specified duration."""
        # Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp())
        frames_needed = int(duration * self.fps)
        logger.info(f"Recording {frames_needed} frames at {self.fps} FPS")

        try:
            frame_files = []
            for frame_num in range(frames_needed):
                start_time = time.time()

                # Capture browser screenshot
                screenshot = self.driver.get_screenshot_as_base64()

                # Save frame as PNG
                frame_path = temp_dir / f"frame_{frame_num:06d}.png"
                with open(frame_path, 'wb') as f:
                    f.write(base64.b64decode(screenshot))
                frame_files.append(frame_path)

                # Calculate sleep time to maintain FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, (1/self.fps) - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Use ffmpeg to combine frames into video
            logger.info("Combining frames into video...")

            # Create ffmpeg input from frame pattern
            stream = ffmpeg.input(str(temp_dir / 'frame_%06d.png'),
                                pattern_type='sequence',
                                framerate=self.fps)

            # Add video codec settings
            stream = ffmpeg.output(stream, output_path,
                                 vcodec='libvpx',
                                 video_bitrate='2M',
                                 **{'deadline': 'realtime', 'cpu-used': 4})

            # Run ffmpeg
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        finally:
            # Cleanup temporary files
            for frame_file in frame_files:
                try:
                    os.unlink(frame_file)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

            logger.info("Recording completed")

    def convert(self, input_path: str, output_path: str):
        """Convert rrweb JSON recording to video."""
        try:
            # Read and parse input file
            logger.info(f"Reading input file: {input_path}")
            with open(input_path, 'r') as f:
                file_content = json.load(f)

            # Extract events
            if not file_content.get('data') or not file_content['data'].get('snapshots'):
                raise ValueError("Invalid file format: expected data.snapshots array")

            events = file_content['data']['snapshots']
            logger.info(f"Found {len(events)} events")

            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                self.temp_html = f.name
                f.write(self.create_html_player(events))

            # Setup and start browser
            self.setup_driver()

            # Load the page
            logger.info("Loading page in browser")
            self.driver.get(f"file://{self.temp_html}")

            # Wait for player to be ready
            WebDriverWait(self.driver, 20).until(
                lambda d: d.execute_script("return window.playerReady === true")
            )
            logger.info("Player ready")

            # Calculate duration
            duration = (events[-1]['timestamp'] - events[0]['timestamp']) / 1000
            logger.info(f"Recording duration: {duration} seconds")

            # Start recording
            logger.info("Starting recording")
            self.record_browser(duration + 2, output_path)  # Add 2 seconds buffer

            logger.info(f"Video saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

        finally:
            # Cleanup
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.driver:
                self.driver.quit()
            if self.temp_html and os.path.exists(self.temp_html):
                os.unlink(self.temp_html)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")