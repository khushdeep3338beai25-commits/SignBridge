# System architecture

1. `getUserMedia()` obtains a local webcam stream.
2. MediaPipe Hand Landmarker returns 21 normalized landmarks.
3. `normalize()` subtracts the wrist position and scales by maximum wrist-relative radius.
4. Optional TensorFlow.js model maps the 63-value feature vector to class probabilities.
5. A configurable threshold and consecutive-prediction stability check gate output.
6. Accepted labels can be added to editable text and spoken using Web Speech API.
7. Session timing, FPS, accepted/rejected counts are displayed separately from offline test metrics.
