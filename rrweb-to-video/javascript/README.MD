# RRWeb to Video Converter

A Node.js utility that converts RRWeb session recordings (JSON format) into video files using Puppeteer and the RRWeb Player.

## Prerequisites

- Node.js (v14 or higher)
- NPM

## Installation

1. Clone this repository
2. Install dependencies:
npm install


## Usage

You can use this utility in two ways:

### 1. Command Line Interface

node run.js <input-json-path.json>

## output file is named <input-json-path.webm>

Example:
node run.js ./recordings/session.json

## output file is ./recordings/session.webm


### 2. Programmatic Usage
const convertRRWebToVideo = require('./convert-rrweb');
async function main() {
await convertRRWebToVideo(
'./recordings/session.json',
'./output/video.webm'
);
}


## Input Format

The input JSON file should contain RRWeb recording data in the following structure:
{
"data": {
"snapshots": [
// RRWeb events array
]
}
}


## Output

The converter generates a WebM video file at the specified output path. The video resolution is set to 1920x1080 pixels with 30 FPS.

## Error Handling

The converter includes error handling for:
- Invalid input file format
- Missing snapshots
- Recording process failures
- Cleanup failures

## License

MIT
