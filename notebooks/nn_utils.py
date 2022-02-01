import gc

import numpy as np
from tqdm.notebook import tqdm
import faiss

def get_faiss_index(d, p):
    if p == 2:
        index = faiss.IndexFlatL2(d)
    elif p == np.inf:
        index = faiss.IndexFlat(d, faiss.METRIC_Linf)
    elif p == 1:
        index = faiss.IndexFlat(d, faiss.METRIC_L1)
    else:
        raise ValueError("[_get_nearest_oppo_dist]")
    return index


def get_nearest_oppo_dist(X, y, norm, cache_filename=None):
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    p = norm

    if cache_filename is not None and os.path.exists(cache_filename):
        print(f"[nearest_oppo_dist] Using cache: {cache_filename}")
        ret = joblib.load(cache_filename)
    else:
        print(f"[nearest_oppo_dist] cache {cache_filename} don't exist. Calculating...")

        ret = np.inf * np.ones(len(X))
        X = X.astype(np.float32, copy=False)

        for yi in tqdm(np.unique(y), desc="[nearest_oppo_dist]"):
            index = get_faiss_index(X.shape[1], p)
            index.add(X[y!=yi])

            idx = np.where(y==yi)[0]
            D, _ = index.search(X[idx], 1)
            if p == 2:
                D = np.sqrt(D)
            ret[idx] = np.minimum(ret[idx], D[:, 0])

            del index
            gc.collect()

        if cache_filename is not None:
            joblib.dump(ret, cache_filename)

    return ret