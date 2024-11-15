# RRWeb Recording Converter

This tool converts RRWeb JSON recordings into WebM video files.

## Prerequisites

- Python 3.6 or higher
- Required Python packages (install via pip):
  ```bash
  pip install -r requirements.txt
  ```

## Usage

The converter consists of two main files:
- `convert_rrweb.py`: Contains the core conversion logic
- `run.py`: Provides a command-line interface for easy conversion

### Basic Usage

1. Convert a single RRWeb recording:

bash
python run.py path/to/recording.json

The converter will:
- Validate the input JSON file
- Create a WebM video file in the same directory
- Output the path to the generated video file

### Output

The converter will create a WebM video file with the same name as your input file:
- Input: `recording.json`
- Output: `recording.webm`

### Error Handling

The script will exit with an error message if:
- No input file is specified
- The input file doesn't exist
- The input file is not a JSON file
- Any conversion errors occur during processing

## Example
ash
python run.py recordings/session1.json


## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.