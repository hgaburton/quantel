import quantel

det = quantel.Determinant([1,1,1,0,0,0,0],[1,1,0,1,0,0,0])
print(det)

(det1, phase) = det.get_excitation([0,3], True)
print(det1)
print(phase)

print(det1 > det)



