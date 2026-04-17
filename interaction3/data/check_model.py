import json, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix

rows = [json.loads(l) for l in Path('interaction3/data/dataset.jsonl').read_text('utf-8').splitlines() if l.strip()]
X = np.array([r['features'] for r in rows], dtype='float32')
y = [r['label'] for r in rows]

print('=== 클래스별 샘플 수 ===')
for k,v in sorted(Counter(y).items()):
    print(f'  {k}: {v}')

pipe = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv)
print(f'\n=== 5-fold CV 정확도 ===')
print(f'  각 fold: {[round(s,3) for s in scores]}')
print(f'  평균: {scores.mean():.3f}  표준편차: {scores.std():.3f}')

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
labels = sorted(set(y))
cm = confusion_matrix(y_te, y_pred, labels=labels)
print(f'\n=== 혼동 행렬 (행=실제, 열=예측) ===')
print('         ' + '  '.join(f'{l[:7]:>7}' for l in labels))
for i, row in enumerate(cm):
    print(f'  {labels[i][:7]:>7}  ' + '  '.join(f'{v:>7}' for v in row))
