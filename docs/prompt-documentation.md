# Prompt and development documentation

## Original project requirements
The supplied brief requested SignBridge AI: a webcam-based, local-first hand-sign prototype with MediaPipe landmarks, a custom TensorFlow/TensorFlow.js classifier, confidence and stability checks, sentence construction, speech synthesis, dataset collection, evaluation, unknown-sign rejection, responsive UI, and tutor-focused documentation.

## Development prompt
Build a complete, runnable SignBridge AI project for a university AI/ML assignment. Prioritize genuine camera processing, visible 21-point hand landmarks, consistent normalized features, real dataset collection/export, a reproducible training pipeline, optional trained-model loading, speech and text tools, measured runtime metrics, responsible limitations, and a polished responsive interface. Do not fabricate a trained model or evaluation results.

## Implementation decisions
- Vite + vanilla JavaScript keeps the frontend approachable.
- MediaPipe Tasks Vision performs local hand landmark detection.
- Features are wrist-relative and radius-normalized, giving 63 numeric inputs.
- Browser-collected samples are stored in localStorage and exportable.
- TensorFlow.js model loading is explicit because no labeled dataset/model was supplied.
- Python training produces a compact MLP and real held-out evaluation artifacts.
- Live timing/FPS are distinguished from classifier accuracy.

## Known limitations / follow-up
- A trained model must be created from real samples; none is included.
- Dynamic signs need temporal sequence modeling.
- Softmax thresholding is only a limited rejection method.
- Validate camera/browser compatibility and TensorFlow.js conversion in the target environment.
- Define a specific sign language and precise class definitions before collecting a serious dataset.

## Testing performed
The project structure and source files are included. Run `npm install`, `npm run build`, and the checklist in README.md on the target laptop. Hardware/browser camera behavior must be verified on the user's machine.
