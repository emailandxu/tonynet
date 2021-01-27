import numpy as np

def calc_wer(ref, hypo):
    """
    ref: ["你", "好"]
    hypo: ["你","掉"]
    return: error amount of substitutions, insertions, deletions
    """
    word_len = len(ref)

    errors = np.zeros([word_len + 1, len(hypo) + 1, 3])
    errors[0, :, 1] = np.arange(len(hypo) + 1)
    errors[:, 0, 2] = np.arange(word_len + 1)
    substitution = np.array([1, 0, 0])
    insertion = np.array([0, 1, 0])
    deletion = np.array([0, 0, 1])
    for r, ref in enumerate(ref):
        for d, dec in enumerate(hypo):
            errors[r + 1, d + 1] = min((
                errors[r, d] + (ref != dec) * substitution,
                errors[r + 1, d] + insertion,
                errors[r, d + 1] + deletion), key=np.sum)

    s, i, d = tuple(errors[-1, -1])
    return s, i, d, word_len, (s + i + d)/word_len

if __name__ == "__main__":
    pass

