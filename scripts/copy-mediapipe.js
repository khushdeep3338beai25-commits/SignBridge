import { cp, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const source = resolve('node_modules/@mediapipe/tasks-vision/wasm');
const destination = resolve('public/mediapipe/wasm');
if (!existsSync(source)) {
  console.error('MediaPipe WASM files were not found. Run `npm install` first.');
  process.exit(1);
}
await mkdir(resolve('public/mediapipe'), { recursive: true });
await rm(destination, { recursive: true, force: true });
await cp(source, destination, { recursive: true });
console.log('MediaPipe WASM files copied to public/mediapipe/wasm');
