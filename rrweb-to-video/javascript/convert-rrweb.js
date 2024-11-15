const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function convertRRWebToVideo(jsonPath, outputPath) {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // Set viewport size
  await page.setViewport({
    width: 1920,
    height: 1080
  });

  // Read and parse the JSON file to extract events
  const fileContent = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

  // Debug logging
  console.log('File structure:', Object.keys(fileContent));
  if (fileContent.data) {
    console.log('Data structure:', Object.keys(fileContent.data));
  }

  // Check if we have the expected structure
  if (!fileContent.data || !fileContent.data.snapshots) {
    console.error('Invalid structure:', JSON.stringify(fileContent, null, 2).slice(0, 500) + '...');
    throw new Error('Invalid file format: expected data.snapshots array');
  }

  const events = fileContent.data.snapshots;
  console.log('Number of snapshots found:', events.length);

  // ... existing code ...

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <title>RRWeb Playback</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/style.css" />
        <script src="https://cdn.jsdelivr.net/npm/rrweb-player@latest/dist/index.js"></script>
        <style>
          html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
          }
          #player {
            width: 100vw;
            height: 100vh;
          }
          .rr-player {
            width: 100% !important;
            height: 100% !important;
          }
          .rr-player__frame {
            width: 100% !important;
            height: calc(100% - 80px) !important;
          }
        </style>
      </head>
      <body>
        <div id="player"></div>
        <script>
          window.addEventListener('load', () => {
            const events = ${JSON.stringify(events)};
            const replayer = new rrwebPlayer({
              target: document.getElementById('player'),
              data: {
                events,
                autoPlay: true,
                width: window.innerWidth,
                height: window.innerHeight,
              }
            });
            window.playerReady = true;
          });
        </script>
      </body>
    </html>
  `;

// ... existing code ...

  // Write temporary HTML file
  const tempHtml = path.join(__dirname, 'temp.html');
  fs.writeFileSync(tempHtml, html);

  try {
    // Navigate to the page with simpler loading strategy
    await page.goto(`file://${tempHtml}`, {
      waitUntil: 'load',
      timeout: 120000 // 2 minutes timeout
    });

    // Wait for player to be ready
    await page.waitForFunction('window.playerReady === true', { timeout: 120000 });
    console.log('Page and player loaded successfully');

    // Start recording
    await page.screencast({
      path: outputPath,
      fps: 30,
    });

    // Wait for replay to finish
    if (!events || !events.length) {
      throw new Error('No snapshots found in the input file');
    }
    const duration = (events[events.length - 1].timestamp - events[0].timestamp) / 1000;
    console.log('Calculated duration:', duration, 'seconds');

    // Use setTimeout instead of waitForTimeout
    await new Promise(resolve => setTimeout(resolve, duration * 1000 + 2000)); // Added extra buffer

    // Stop recording before closing
    console.log('Stopping recording...');
    await page.screencast({ path: null });
    console.log('Recording stopped');

  } catch (error) {
    console.error('Error during conversion:', error);
    throw error;
  } finally {
    // Cleanup
    try {
      await browser.close();
      if (fs.existsSync(tempHtml)) {
        fs.unlinkSync(tempHtml);
      }
    } catch (cleanupError) {
      console.error('Error during cleanup:', cleanupError);
    }
  }

  console.log(`Video saved to: ${outputPath}`);
}

module.exports = convertRRWebToVideo;