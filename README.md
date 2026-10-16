# 🤟 SignBridge AI — Breaking Communication Barriers with Real-Time Edge AI

**SignBridge AI** is a browser-based, AI-powered prototype designed to make communication more accessible by recognizing predefined hand gestures and converting them into text and speech in real time.

The project uses **MediaPipe, Edge AI concepts, and browser-based machine learning** to demonstrate webcam hand tracking, landmark visualization, gesture recognition, labeled dataset collection, and sign-to-text/speech workflows.

> **Our mission:** To explore how accessible, privacy-conscious AI can help bridge communication gaps between people who use sign language and those who do not understand it.

---

## ✨ Key Features

- 📷 **Real-Time Webcam Tracking** — Start, stop, and restart the webcam with camera permission handling and resource cleanup.
- 🤖 **AI-Powered Gesture Recognition** — Uses MediaPipe Tasks Vision and a pretrained gesture-recognition model for supported static hand poses.
- 🖐️ **21-Point Hand Landmark Detection** — Visualizes hand landmarks and the skeletal connections between them.
- 📐 **Landmark Normalization** — Converts detected hand landmarks into 63 normalized coordinate features.
- 🗂️ **Custom Dataset Collection** — Collect labeled gesture samples directly in the browser.
- 📤 **Dataset Export** — Export collected samples in CSV and JSON formats for further processing and training.
- 🧠 **Custom Model Training Workflow** — Supports an optional TensorFlow.js classifier and a Python-based model-training pipeline.
- 💬 **Gesture-to-Text Conversion** — Maps supported gestures to configurable demonstration labels.
- 🔊 **Text-to-Speech** — Uses the browser's Web Speech API to speak recognized labels and composed messages.
- 🎯 **Confidence Threshold** — Helps reject predictions below a configurable confidence level.
- ✍️ **Editable Messages** — Edit recognized text and build short messages before speaking.
- 📊 **Performance Dashboard** — Displays live tracking/inference timing and FPS information.
- 🔒 **Local-First Processing** — Camera frames are processed in the browser by default rather than uploaded to a project server.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| HTML, CSS, JavaScript | User interface and browser interactions |
| React and Vite | Frontend development and local development server |
| MediaPipe Tasks Vision | Hand landmark detection and pretrained gesture recognition |
| TensorFlow.js | Optional custom gesture-classification workflow |
| Python | Dataset processing and model training |
| NumPy and scikit-learn | Numerical processing and classification evaluation in the training pipeline |
| WebRTC / MediaDevices API | Webcam access |
| Web Speech API | Text-to-speech output |
| LocalStorage | Temporary persistence of collected samples |
| CSV and JSON | Dataset export and exchange |

*The exact training libraries and dependencies depend on the packages configured in `training/requirements.txt` or the project's training environment.*

---

## 🧠 How SignBridge AI Works

The application follows a browser-based computer vision pipeline.

1. **Capture:** The browser requests permission to access the webcam.
2. **Detection:** MediaPipe detects the hand and identifies 21 landmarks.
3. **Preprocessing:** The landmark coordinates are normalized into a 63-value feature vector.
4. **Recognition:** The pretrained gesture recognizer identifies supported gestures. An optional custom classifier can be used for a trained vocabulary.
5. **Filtering:** A confidence threshold and stability filtering help reduce unreliable or repeated predictions.
6. **Text Generation:** The recognized gesture is mapped to a configured demonstration label.
7. **Speech Output:** The Web Speech API converts the composed text into spoken audio.

### Architecture

```text
Webcam Input
     ↓
MediaPipe Hand Detection
     ↓
21 Hand Landmarks
     ↓
Wrist-Relative Normalization
     ↓
63 Numerical Features
     ↓
Gesture Recognition
     ↓
Confidence & Stability Filtering
     ↓
Gesture-to-Text Mapping
     ↓
Editable Message
     ↓
Text-to-Speech Output
```

**Edge AI approach:** Core webcam processing and supported gesture inference run in the browser. The application is designed to reduce dependence on a remote inference server.

---

## ✋ Supported Gestures

The current prototype uses a pretrained MediaPipe gesture model to recognize seven supported static poses.

| Recognized Pose | Demonstration Label |
|---|---|
| Open Palm | HELLO |
| Closed Fist | STOP |
| Pointing Up | HELP |
| Thumb Up | YES |
| Thumb Down | NO |
| Victory (V) | PEACE |
| I Love You | I LOVE YOU |

These mappings are demonstration labels for the prototype. They are **not validated translations of American Sign Language (ASL) or Indian Sign Language (ISL)**.

The interface also provides example vocabulary labels such as `THANK YOU`, `PLEASE`, and `WATER` for dataset collection. Their presence does not mean the pretrained model already recognizes those signs.

---

## 📐 Landmark Feature Preprocessing

SignBridge AI represents a detected hand using 21 landmarks, with each landmark containing three coordinates: \(x\), \(y\), and \(z\).

This produces:

`21 landmarks × 3 coordinates = 63 features`

The preprocessing procedure is:

1. Select the wrist as the reference landmark (landmark 0).
2. Subtract the wrist coordinates from every landmark.
3. Calculate the maximum Euclidean distance from the wrist.
4. Divide the shifted coordinates by that maximum distance, when nonzero.
5. Flatten the normalized coordinates into a 63-value feature vector.

This makes the representation less sensitive to hand position and scale. The same preprocessing must be applied to training and inference data for a custom classifier to work correctly.

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18 or later
- npm
- A supported modern browser
- A working webcam
- Camera permission enabled
- Internet access on the first run if MediaPipe assets are fetched from hosted URLs

### Installation

**1. Clone the repository**

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and project folder name.

**2. Install dependencies**

```bash
npm install
```

**3. Start the development server**

```bash
npm run dev
```

**4. Open the application**

Open the exact localhost URL printed by Vite, commonly:

`http://localhost:5173`

Allow camera access when prompted.

> Do not open `index.html` directly using a `file://` URL. Browser camera APIs generally require localhost or a secure HTTPS context.

---

## 📷 Camera and MediaPipe Setup

The project includes a MediaPipe WebAssembly asset setup intended to avoid broken CDN paths and incorrect MIME types.

Before starting the application, make sure the MediaPipe WASM files are available under:

```text
public/
└── mediapipe/
    └── wasm/
```

Install dependencies and run:

```bash
npm install
npm run dev
```

If the project includes a setup script that copies WASM assets from the installed npm package, run the configured script before launching Vite. Check `package.json` for the actual script name.

### Troubleshooting Camera Errors

If the application reports that the camera could not start:

1. Open the application through the localhost URL printed by Vite.
2. In Safari, open **Safari → Settings → Websites → Camera**.
3. Set the permission for localhost to **Allow**.
4. Close other applications that may be using the webcam, such as FaceTime, Zoom, or Google Meet.
5. Reload the application and retry camera access.
6. Check the browser console for MediaPipe, camera-permission, or asset-loading errors.

The application should handle camera startup failures gracefully and display a readable error message instead of an unhelpful error such as `[object Event]`.

---

## 🗃️ Dataset Collection

SignBridge AI includes an in-browser workflow for collecting labeled hand-landmark samples.

### Collection workflow

1. Start the webcam.
2. Select a label for the intended gesture.
3. Position the hand clearly within the camera frame.
4. Capture multiple samples.
5. Repeat under different lighting conditions, hand positions, and recording sessions.
6. Export the collected dataset as CSV or JSON.
7. Review and preprocess the dataset before training.

### Recommended dataset practices

- Collect examples from multiple recording sessions.
- Include natural variations in lighting, camera distance, and hand orientation.
- Avoid relying entirely on nearly identical consecutive frames.
- Use a consistent, documented definition for each gesture.
- Obtain appropriate consent when collecting data from other participants.
- Keep training and evaluation data separate.

Collected samples are stored in browser LocalStorage until exported or cleared. Clearing browser data may remove samples that have not been backed up.

---

## 🧪 Training a Custom Model

The pretrained MediaPipe model recognizes its supported gestures. To add a new vocabulary, a separate custom classifier must be trained using labeled data.

The project includes a Python training workflow that can generate held-out classification metrics and a confusion matrix from supplied data.

For instructions, see [`training/README.md`](training/README.md).

### Suggested training workflow

1. Collect and export labeled samples.
2. Validate labels and remove invalid or duplicate records.
3. Split the dataset into training and held-out evaluation sets.
4. Train the classifier using the supplied training script.
5. Evaluate the model on data it has not seen during training.
6. Inspect the classification report and confusion matrix.
7. Export the trained model in a format supported by the application.
8. Load the model and verify its label mapping and preprocessing.

**Important:** Use participant-level or recording-session-level splits where possible. Randomly splitting nearly identical frames across training and test sets can produce misleadingly high evaluation scores.

The exact model architecture, export procedure, and supported model format should follow the implementation in the `training/` directory and the application's model loader.

---

## 🔐 Privacy and Responsible AI

SignBridge AI follows a local-first design for webcam processing.

- Camera frames are processed in the browser by default.
- Collected landmark samples are stored locally until exported or cleared.
- Users must grant camera permission before webcam access.
- The prototype does not claim to provide complete sign-language translation.
- Custom datasets should be collected and used responsibly, with appropriate consent.

Local processing of camera frames does not automatically mean that every external asset request or browser feature is offline. The initial MediaPipe asset download may require internet access.

---

## ⚠️ Limitations

- **Limited vocabulary:** The current pretrained workflow supports seven specified static gesture categories.
- **Not a complete translator:** Recognizing isolated hand poses is different from translating continuous sign language.
- **Language-specific meanings:** ASL, ISL, and other sign languages have distinct vocabularies and grammatical structures.
- **Movement-based signs:** Some signs require temporal information rather than a single frame.
- **Facial expressions and context:** Meaning may depend on facial expressions, body movements, grammar, and context.
- **Variable accuracy:** Lighting, occlusion, camera quality, hand orientation, and individual differences can affect recognition.
- **Confidence limitations:** A high softmax confidence score does not guarantee a correct prediction or reliably detect unknown gestures.
- **Prototype mappings:** Gesture-to-word mappings require validation with fluent signers before real-world use.

Do not use this prototype as a substitute for a qualified interpreter in medical, legal, emergency, or other critical communication.

---

## 🎓 Tutor Demonstration Plan

A short demonstration can show the complete project workflow:

1. Start the webcam and demonstrate live hand landmark tracking.
2. Show the seven supported pretrained gesture categories.
3. Collect labeled samples for a custom gesture vocabulary.
4. Export the dataset and explain the 63-feature representation.
5. Train a custom classifier using the supplied training pipeline.
6. Show the held-out classification report and confusion matrix.
7. Demonstrate prediction filtering, message editing, and speech output.
8. Explain the limitations and possible future improvements.

### What makes SignBridge AI different?

Unlike a basic alphabet-detection demonstration, SignBridge AI is designed to demonstrate a broader workflow that includes:

- Real-time hand landmark extraction.
- Normalized feature generation.
- Browser-based sample collection and dataset export.
- A pathway for training and evaluating a custom classifier.
- Gesture-to-text and text-to-speech interaction.
- Performance monitoring and privacy-conscious processing.

The distinction is the **end-to-end prototype workflow**, rather than a claim that the project already translates an entire sign language.

---

## 🗺️ Future Improvements

- Support for a larger, carefully validated gesture vocabulary.
- Integration of Indian Sign Language-specific datasets and expert-reviewed labels.
- Temporal modeling for motion-based signs.
- Improved prediction smoothing and unknown-gesture rejection.
- More comprehensive evaluation across signers and recording environments.
- Offline asset support and improved low-connectivity operation.
- Accessibility improvements for keyboard navigation, screen readers, and visual feedback.
- User testing with Deaf and hard-of-hearing communities and fluent signers.

---

## ✅ Testing Checklist

- [ ] Allow and deny camera permissions and verify error handling.
- [ ] Start, stop, and restart the webcam.
- [ ] Verify landmark overlay alignment at different viewport sizes.
- [ ] Capture, export, reload, and clear sample data.
- [ ] Confirm that exported labels match the selected vocabulary.
- [ ] Train the custom classifier using multiple recording sessions.
- [ ] Inspect held-out evaluation metrics and the confusion matrix.
- [ ] Load the trained model and verify its label mapping.
- [ ] Test low-confidence predictions and repeated gestures.
- [ ] Test sentence editing and speech controls.
- [ ] Confirm that demonstration labels are not presented as validated sign-language translations.
- [ ] Ensure performance metrics are not presented as model accuracy without proper evaluation.

---

## 📁 Project Structure

The following is an illustrative structure. Actual filenames may vary depending on the repository implementation.

```text
SignBridge-AI/
├── public/
│   └── mediapipe/
│       └── wasm/
├── src/
│   ├── components/
│   ├── ...
│   └── ...
├── training/
│   ├── README.md
│   └── ...
├── package.json
├── package-lock.json
├── index.html
└── README.md
```

---

## 🌟 Project Goal

SignBridge AI explores how real-time computer vision and browser-based machine learning can contribute to more accessible communication.

By combining gesture recognition, dataset collection, custom model training, text generation, and speech synthesis, the project provides a foundation for further research into assistive communication technologies.

**Built with a focus on accessibility, practical AI learning, and responsible innovation.**

