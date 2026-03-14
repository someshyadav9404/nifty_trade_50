import numpy as np
preds = np.load('nifty_trade_50/artifacts/paper1/preds.npy')
print('Unique signals:', np.unique(preds))
print('First 20 signals:', preds[:20])
print('Count of 1s:', np.sum(preds==1))
print('Count of -1s:', np.sum(preds==-1))
print('Count of 0s:', np.sum(preds==0))

