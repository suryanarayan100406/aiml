/**
 * pixi-live2d-display CDN Downloader
 * Downloads the required libraries locally to avoid COEP/CORS issues.
 */
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const vendorDir = path.join(__dirname, 'frontend', 'vendor');
if (!fs.existsSync(vendorDir)) fs.mkdirSync(vendorDir, { recursive: true });

const files = [
    {
        url: 'https://cdnjs.cloudflare.com/ajax/libs/pixi.js/7.4.0/pixi.min.js',
        dest: path.join(vendorDir, 'pixi.min.js')
    },
    {
        url: 'https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js',
        dest: path.join(vendorDir, 'live2dcubismcore.min.js')
    },
    {
        url: 'https://cdn.jsdelivr.net/npm/pixi-live2d-display/dist/cubism4.min.js',
        dest: path.join(vendorDir, 'cubism4.min.js')
    }
];

function download(url, dest) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(dest);
        const protocol = url.startsWith('https') ? https : http;
        
        protocol.get(url, (response) => {
            // Handle redirects
            if (response.statusCode === 301 || response.statusCode === 302) {
                download(response.headers.location, dest).then(resolve).catch(reject);
                return;
            }
            
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                const size = fs.statSync(dest).size;
                console.log(`  ✅ ${path.basename(dest)} (${(size/1024).toFixed(0)} KB)`);
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(dest, () => {});
            reject(err);
        });
    });
}

async function main() {
    console.log('📦 Downloading Live2D vendor libraries...\n');
    
    for (const f of files) {
        console.log(`  ⏳ Downloading ${path.basename(f.dest)}...`);
        try {
            await download(f.url, f.dest);
        } catch (e) {
            console.error(`  ❌ Failed: ${e.message}`);
        }
    }
    
    console.log('\n✅ Done! Libraries saved to frontend/vendor/');
}

main();
