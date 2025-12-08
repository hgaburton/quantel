import numpy as np
from quantel import FourArray

d = 5
np.random.seed(7)
arr = np.random.rand(d,d,d,d)

# Apply symmetry to give s4 symmetry [p,q,r,s] = [p,s,r,q] = [r,q,p,s] = [r,s,p,q]
for p in range(d):
    for q in range(d):
        for r in range(d):
            for s in range(q+1, d):
                avg = 0.25 * (arr[p,q,r,s] + arr[p,s,r,q] + arr[r,q,p,s] + arr[r,s,p,q])
                arr[p,q,r,s] = avg
                arr[p,s,r,q] = avg
                arr[r,q,p,s] = avg
                arr[r,s,p,q] = avg

print(arr)
four_array = FourArray(arr, d, d, d, d)

print("Element (2,3,1,4): ", four_array(2,3,1,4), " Expected: ", arr[2,3,1,4])
new_arr = four_array.array()
print(np.linalg.norm(new_arr - arr))
