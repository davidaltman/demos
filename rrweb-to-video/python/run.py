import sys
from pathlib import Path
from convert_rrweb import RRWebConverter
import logging

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python run.py <path-to-json-file>")
        sys.exit(1)

    # Get input file path
    input_file = Path(sys.argv[1])

    # Validate input file
    if not input_file.exists():
        print(f"Input file does not exist: {input_file}")
        sys.exit(1)

    if input_file.suffix != '.json':
        print(f"Input file must be a JSON file: {input_file}")
        sys.exit(1)

    # Create output path
    output_file = input_file.with_suffix('.webm')

    try:
        # Initialize converter and run conversion
        converter = RRWebConverter()
        converter.convert(str(input_file), str(output_file))
        print(f"Conversion completed successfully: {output_file}")

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()