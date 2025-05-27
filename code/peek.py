from scipy.io import loadmat

data = loadmat("data/example1.mat", squeeze_me=True, struct_as_record=False)
print(data.keys())
print(data['__header__'])
print(data['__version__'])
print(data['__globals__'])
print(data['A'].shape)
print(data['y'].shape)
print(type(data['y'][0]))
print(data['y'][:20])


