import h5py
filename = 'data/Challenge2_1_train.h5'

f = h5py.File(filename, 'r+')

print("Keys before deletion:", list(f.keys()))

for key in list(f.keys()):
    # perform a case-insensitive search by comparing lowercase versions
    kl = key.lower()
    if any(term in kl for term in ("tke", "rms", "weight")):
        del f[key]

print("Keys after deletion:", list(f.keys()))