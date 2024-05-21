import quantel

detlist = []
detlist.append(quantel.Determinant([1,1,1,0,0,0,0],[1,1,0,1,0,0,0]))
detlist.append(quantel.Determinant([1,1,1,0,0,0,0],[1,1,1,0,0,0,0]))
detlist.append(quantel.Determinant([1,1,0,1,0,0,0],[1,1,1,0,0,0,0]))
coeff = [1.,0.,0.]

ci = quantel.CIexpansion(detlist, coeff)
print("PRINT")
ci.print(-1)
