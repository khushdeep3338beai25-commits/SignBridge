# Evaluation protocol

Do not report accuracy until labeled test data exists. Collect multiple sessions and, where possible, multiple participants. Keep test sessions separate from training sessions. The training script emits a classification report and confusion matrix from its held-out split. Report class-wise precision, recall, F1, support, dataset size, split method, and limitations.

For threshold experiments, evaluate the same held-out examples at each threshold. Report coverage/acceptance rate, error rate among accepted examples, and rejection rate. A higher threshold may reduce accepted errors while rejecting more valid signs.
