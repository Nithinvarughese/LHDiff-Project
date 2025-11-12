import hashlib

def simhash(text):
    tokens = text.split()
    hash_bits = 64
    v = [0] * hash_bits

    for token in tokens:
        h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
        for i in range(hash_bits):
            bitmask = 1 << i
            if h & bitmask:
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(hash_bits):
        if v[i] >= 0:
            fingerprint |= 1 << i

    return fingerprint


def hamming_distance(hash1, hash2):
    x = hash1 ^ hash2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist


def generate_candidates(old_lines, new_lines, k=15):
    candidates = {}
    simhash_old = [simhash(line) for line in old_lines]
    simhash_new = [simhash(line) for line in new_lines]

    for i, old_hash in enumerate(simhash_old):
        distances = []
        for j, new_hash in enumerate(simhash_new):
            dist = hamming_distance(old_hash, new_hash)
            distances.append((j, dist))

        distances.sort(key=lambda x: x[1])
        candidates[i] = [idx for idx, _ in distances[:k]]

    return candidates


if __name__ == "__main__":
    old_file = open("old.txt").read().splitlines()
    new_file = open("new.txt").read().splitlines()

    candidate_map = generate_candidates(old_file, new_file)
    for old_idx, candidates in candidate_map.items():
        print(f"Old line {old_idx+1} → possible matches: {[c+1 for c in candidates]}")
