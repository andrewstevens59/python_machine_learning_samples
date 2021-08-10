import argparse
import gzip
import os
from logging import getLogger
from logging.config import fileConfig
from multiprocessing import Pool
from struct import pack, unpack

import numpy as np
from scipy.sparse import csr_matrix

logger = getLogger(__name__)
fileConfig('../conf/logging.conf', disable_existing_loggers=False)

def load_weights(weights_file):
    with gzip.open(weights_file) as f, open(weights_file + 'ids', 'wb') as id_file, open(weights_file + 'weights', 'wb') as w_file:
        num_weights = 0
        for line in f:
            idx, w = line.split(':')
            id_file.write(pack('i', int(idx)))
            w_file.write(pack('f', float(w)))
            num_weights += 1
    return num_weights

def load_npz(npz_file, line_start):
    logger.info("loading validation dataset")
    npz = np.load(npz_file)
    m = csr_matrix((npz['data'], npz['indices'], npz['indptr']), shape=npz['shape'])
    return m[line_start:]

def load_from_bin(weight_paths, num_weights):
    ret = None
    for i, weight_path in enumerate(weight_paths):
        id_file = weight_path+'ids'
        w_file = weight_path+'weights'
        ids = unpack(str(num_weights)+'i', open(id_file, 'rb').read())
        if ret is None:
            ret = np.zeros((ids[-1]+1, len(weight_paths)), dtype=float)
        weights = unpack(str(num_weights)+'f', open(w_file, 'rb').read())
        for j, w in zip(ids, weights):
            ret[j][i] = w
    return ret

def quad(ids, width):
    l = []
    l.extend(ids)
    for i in range(len(ids)):
        x = ids[i]
        base = (2 * width - x) * (x + 1) / 2 - x
        l.extend([base+y for y in ids[i+1:]])
    l.append(width * (width + 1) / 2)
    return l

def to_probability(sum_weights, negative_weight):
    p = 1 / (1 + np.exp(-sum_weights))
    actp = p / (p + (1 - p) * negative_weight)
    return actp


def predict(matrix, weight_paths, num_weights, negative_weight):
    try:
        weights = load_from_bin(weight_paths, num_weights)
        num_rows, num_cols = matrix.shape
        ps = np.zeros((num_rows, weights.shape[1]), dtype=float)

        cnt = 0
        for row in range(num_rows):
            ids = quad(matrix[row].indices, num_cols)
            ps[row] = to_probability(np.sum(weights[ids], axis=0), negative_weight)
            cnt += 1

        return ps
    except:
        logger.exception("Exception in predict()")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions for a given input set of weights')
    parser.add_argument('weights', nargs='+', help='1 or more weight files')
    parser.add_argument("--starting", type=int, default=0, help="The starting line of the score file. 0 index")
    parser.add_argument("--neg-sample-rate", type=float, default=1.0, help="The negative downsampling rate to account for")
    parser.add_argument('-p', '--parallelism', dest='parallelism', type=int, default=4)
    parser.add_argument('-o', '--output', dest='out_path', type=str, default='prediction')
    args = parser.parse_args()

    min_parallelism = 1
    max_parallelism = 8
    parallelism = max(min(args.parallelism, max_parallelism), min_parallelism)
    logger.info("parallelism: %d" % parallelism)

    pool = Pool(parallelism, maxtasksperchild=1)

    weights = args.weights
    negative_weight = 1.0 / args.neg_sample_rate

    logger.info("loading weights ... " + str(weights))
    result = pool.map(load_weights, weights)
    num_weights = result[0]
    logger.info("loaded %d weights", num_weights)

    matrix = load_npz('matrix_data.npz', args.starting)
    num_rows = matrix.shape[0]
    logger.info("matrix has shape %s", str(matrix.shape))
    jobs = []
    logger.info("start scoring")
    for i in range(parallelism):
        start = i * num_rows / parallelism
        end = (i+1) * num_rows / parallelism
        local_matrix = matrix[start:end]
        jobs.append(pool.apply_async(func=predict, args=(local_matrix, weights, num_weights, negative_weight)))
    pool.close()
    pool.join()
    result = [job.get() for job in jobs]
    logger.info("finished scoring")
    all_preds = np.concatenate(result)
    for i, path in enumerate(weights):
        weight_fname = os.path.basename(path)
        out_path = (args.out_path + '.'+ weight_fname.split('.')[2] if len(weight_fname.split('.')) == 4 else args.out_path) + '.dat'
        logger.info('writing predictions to ' + out_path)
        preds = all_preds[:, i]
        with open(out_path, 'w') as of:
            of.write('\n'.join(map(str, preds)))
    logger.info('finished writing predictions')
