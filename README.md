# SignBridge AI — Breaking Communication Barriers with Real-Time Edge AI

A local-first browser prototype for webcam hand tracking, landmark visualization, labeled data collection, and sign-to-text/speech workflows.

## What works in this delivery
- Real webcam permission flow, start/stop and resource cleanup.
- MediaPipe Tasks Vision Gesture Recognizer with a built-in pretrained gesture model and 21-point skeleton overlay.
- Wrist-relative landmark normalization (63 values).
- In-browser labeled sample collection saved to local storage; CSV/JSON export and clear controls.
- Built-in recognition for seven common static hand poses, gesture-to-demo-label mapping, confidence threshold, sentence editing, browser speech synthesis, plus optional custom TensorFlow.js model loading.
- Live tracking/inference timing and FPS dashboard.
- Python model-training script that generates real reports from supplied data.

## Important model status
**A pretrained MediaPipe gesture model is used for the seven supported poses**: Open Palm, Closed Fist, Pointing Up, Thumb Up, Thumb Down, Victory (V), and I Love You. Demo mappings are HELLO, STOP, HELP, YES, NO, PEACE, and I LOVE YOU. These mappings are convenient prototype labels, not a validated translation of ASL or Indian Sign Language. Custom vocabulary still requires collecting data and training a custom classifier.

## Run locally
Requires Node.js 18+.
```bash
npm install
npm run dev
```
Open the localhost URL printed by Vite and allow camera access. Internet is needed on first run to load MediaPipe assets from their hosted URLs. Browser camera APIs generally require localhost or HTTPS.

## Sign vocabulary
The UI begins with example labels: HELLO, HELP, YES, NO, THANK YOU, PLEASE, STOP, WATER, I LOVE YOU. These are class names only—not a claim that their poses are universal signs. Define the intended sign language and document each pose before collecting data. Start with static poses; motion-based signs need temporal features.

## Architecture
Webcam → MediaPipe Hand Landmarker → 21 landmarks → wrist-relative normalization → optional TensorFlow.js MLP → confidence threshold + stability filter → editable text → Web Speech API.
Frames are processed in the browser. Training samples are stored in localStorage until exported or cleared.

## Feature preprocessing
For each landmark, subtract the wrist (landmark 0) x/y/z. Compute the maximum Euclidean distance from the wrist across landmarks and divide every shifted coordinate by that radius. This yields 21 × 3 = 63 values. Use the same preprocessing for training and inference.

## Train a model
See [training/README.md](training/README.md). The Python script creates actual held-out classification metrics and a confusion matrix. Use diverse recordings and preferably participant/session-level splits. A random split of near-identical frames can inflate scores.

## Limitations and responsible use
- Static limited-vocabulary prototype, not complete sign-language translation.
- Sign languages differ in vocabulary and grammar; many signs use movement, facial expression, and context.
- Softmax confidence is not a reliable guarantee of unknown-class detection.
- Performance varies with lighting, camera, pose, and signer.
- Do not use in place of a qualified interpreter for critical communication.
- Camera processing is local by default; the app does not upload camera frames.

## Tutor demo plan
1. Start camera and show landmark overlay.
2. Collect labeled examples across several sessions.
3. Train and load the model; show confidence and stability behavior.
4. Build and speak a short message.
5. Present the generated test report and confusion matrix; explain the dataset size and limitations.

## Testing checklist
- [ ] Allow/deny camera permission and confirm helpful error.
- [ ] Start, stop, and restart camera.
- [ ] Check overlay alignment at multiple viewport sizes.
- [ ] Capture, export, reload, and clear sample data.
- [ ] Train using multiple sessions and inspect held-out report.
- [ ] Load exported model and confirm label mapping.
- [ ] Test low-confidence rejection and repeated-pose behavior.
- [ ] Test speech controls and sentence editing.
- [ ] Confirm no metrics are represented as classifier accuracy before evaluation.


## Camera troubleshooting
- Run `npm install` and `npm run dev`, then open the exact `http://localhost:5173` URL Vite prints. Do not open `index.html` directly as a `file://` URL.
- In Safari, open **Safari → Settings → Websites → Camera**, set permission for localhost to **Allow**, then reload the page.
- Close other apps that may be using the webcam (FaceTime, Zoom, Meet), then press **Retry camera**.
- The app now retries camera access with simpler constraints when a camera rejects the preferred resolution and displays a clearer error rather than `[object Event]`.

### Camera / MediaPipe loading fix

The MediaPipe WebAssembly runtime is copied from the installed npm package into `public/mediapipe/wasm` before starting Vite or building. This avoids the broken jsDelivr WASM URL that can return 404 or an incorrect MIME type.

```bash
npm install
npm run dev
```

Open the exact localhost URL printed by Vite. If Safari has remembered a denied permission, go to Safari → Settings → Websites → Camera, set localhost to Allow, then reload.
