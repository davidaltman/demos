const converter = require('./convert-rrweb');
const path = require('path');
const fs = require('fs');

// Get input file from command line arguments
const inputFile = process.argv[2];

if (!inputFile) {
    console.error('Please provide an input file path');
    console.error('Usage: node run.js <path-to-json-file>');
    process.exit(1);
}

// Resolve the full path
const fullInputPath = path.resolve(inputFile);

// Create output path by replacing .json with .webm
const outputFile = fullInputPath.replace(/\.json$/, '.webm');

// Debug: Check if file exists
if (!fs.existsSync(fullInputPath)) {
    console.error('Input file does not exist:', fullInputPath);
    process.exit(1);
}

console.log('Input file:', fullInputPath);
console.log('Output file:', outputFile);

// Run the converter
converter(fullInputPath, outputFile)
    .catch(error => {
        console.error('Conversion failed:', error);
        process.exit(1);
    });