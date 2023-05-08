import numpy as np
import time

time.clock = time.time

persontype = [('name', 'S20'), ('age', 'i2'), ('weight', 'f4')]
a = np.array([("Zhang", 32, 75.5), ("Wang", 24, 65.2)], dtype=persontype)

print(a)

start = time.clock()
a[1]
print(a[1])
end = time.clock()
print(( "a[1] cost time: %f" % (end-start)))

start = time.clock()
a['name']
print(a['name'])
end = time.clock()
print( "a['name'] cost time: %f" % (end-start))

start = time.clock()
a[1]['name']
print(a[1]['name'])
end = time.clock()
print( "a[1]['name'] cost time: %f" % (end-start))

start = time.clock()
a['name'][1]
print(a['name'][1])
end = time.clock()
print( "a['name'][1] cost time: %f" % (end-start))

