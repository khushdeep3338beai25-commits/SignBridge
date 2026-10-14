"""Train a compact MLP from SignBridge exported JSON. No metrics are fabricated."""
import argparse, json, os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='datasets/landmarks.json')
    ap.add_argument('--out',default='training/output')
    args=ap.parse_args()
    with open(args.data,encoding='utf8') as f: payload=json.load(f)
    samples=payload.get('samples',payload if isinstance(payload,list) else [])
    if len(samples)<20: raise SystemExit('Collect at least 20 samples; more varied samples per class are strongly recommended.')
    X=np.asarray([s['features'] for s in samples],dtype='float32')
    y=np.asarray([s['label'] for s in samples])
    if X.ndim!=2 or X.shape[1]!=63: raise SystemExit(f'Expected 63 features per sample; got {X.shape}.')
    encoder=LabelEncoder(); yi=encoder.fit_transform(y)
    # Group by recording session when available to reduce near-duplicate leakage.
    groups=np.asarray([s.get('session','ungrouped') for s in samples])
    unique_groups=len(set(groups))
    if unique_groups>=3:
        split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
        train_idx,test_idx=next(split.split(X,yi,groups))
    else:
        rng=np.random.default_rng(42); idx=rng.permutation(len(X)); cut=max(1,int(.8*len(X)))
        train_idx,test_idx=idx[:cut],idx[cut:]
        print('WARNING: session groups are insufficient; random split may leak similar frames.')
    model=tf.keras.Sequential([tf.keras.layers.Input((63,)),tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(64,activation='relu'),
      tf.keras.layers.Dense(len(encoder.classes_),activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    os.makedirs(args.out,exist_ok=True)
    history=model.fit(X[train_idx],yi[train_idx],validation_split=.2,epochs=60,batch_size=32,
      callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)],verbose=1)
    probs=model.predict(X[test_idx],verbose=0); pred=probs.argmax(axis=1)
    report=classification_report(yi[test_idx],pred,labels=np.arange(len(encoder.classes_)),
      target_names=encoder.classes_,zero_division=0,output_dict=True)
    with open(os.path.join(args.out,'classification_report.json'),'w') as f: json.dump(report,f,indent=2)
    cm=confusion_matrix(yi[test_idx],pred,labels=np.arange(len(encoder.classes_)))
    plt.figure(figsize=(8,6)); plt.imshow(cm); plt.xticks(range(len(encoder.classes_)),encoder.classes_,rotation=45,ha='right')
    plt.yticks(range(len(encoder.classes_)),encoder.classes_); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout()
    plt.savefig(os.path.join(args.out,'confusion_matrix.png')); plt.close()
    model.save(os.path.join(args.out,'keras_model.keras'))
    with open(os.path.join(args.out,'metadata.json'),'w') as f:
        json.dump({'labels':encoder.classes_.tolist(),'feature_count':63,'preprocessing':'wrist-relative xyz, divided by maximum Euclidean radius','test_samples':len(test_idx)},f,indent=2)
    with open(os.path.join(args.out,'training_history.json'),'w') as f:
        json.dump({k:[float(v) for v in vals] for k,vals in history.history.items()},f,indent=2)
    print('Held-out report saved. Export model to TensorFlow.js using the instructions in training/README.md.')
if __name__=='__main__': main()
