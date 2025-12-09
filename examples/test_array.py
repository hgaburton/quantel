import numpy as np
import quantel

if __name__ == "__main__":
    # Dimension of arrays
    d = 5
    
    # Test TwoArray
    np.random.seed(7)
    arr = np.random.rand(d,d)
    # Apply symmetry to give s4 symmetry [p,q] = [q,p]
    for p in range(d):
        for q in range(p, d):
            avg = 0.5 * (arr[p,q] + arr[q,p])
            arr[p,q] = avg
            arr[q,p] = avg
    array = quantel.TwoArray(arr)

    print("Element (2,3): ", array(2,3), " Expected: ", arr[2,3])
    new_arr = array.array()
    print("Error for TwoArray = ", np.linalg.norm(new_arr - arr))
    
    
    # Test FourArray
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

    array = quantel.FourArray(arr)

    print("Element (2,3,1,4): ", array(2,3,1,4), " Expected: ", arr[2,3,1,4])
    new_arr = array.array()
    print("Error for FourArray = ", np.linalg.norm(new_arr - arr))

