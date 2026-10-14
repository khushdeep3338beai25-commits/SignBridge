# Training pipeline

1. In the browser, open **AI Training Lab**, collect varied labeled samples, and export JSON.
2. Save the downloaded file as `datasets/landmarks.json`.
3. Create an environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r training/requirements.txt
   python training/train_classifier.py --data datasets/landmarks.json
   ```
4. Training writes a Keras model, metadata, training history, classification report, and confusion matrix under `training/output/`.
5. Install TensorFlow.js converter in a compatible Python environment and convert:
   ```bash
   pip install tensorflowjs
   tensorflowjs_converter --input_format=keras training/output/keras_model.keras public/models/sign-classifier
   ```
6. Copy `training/output/metadata.json` to `public/models/sign-classifier/metadata.json`.
7. Start the Vite app and load `/models/sign-classifier/model.json` in the Training Lab.

## Data and evaluation notes
- The browser preprocessing is wrist-relative xyz coordinates divided by the maximum Euclidean radius, producing 63 values.
- The training script expects those exact features.
- Session-based splitting is used when at least three distinct session identifiers exist. Collect data across multiple sessions for a more meaningful test.
- Small datasets produce unreliable estimates. Do not present the report as general sign-language accuracy.
- This baseline is a static-pose MLP. It does not model motion-based signs.
