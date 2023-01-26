import numpy as np
from itertools import permutations

class FGH:
    # class to store and propagate
    # function value (f), gradient (g) and Hessian (h)
    def __init__(s, f, g, h):
        s.f = f
        s.g = g
        s.h = h

    def __eq__(s, o):
        if (type(s)==type(o)
                and s.f==o.f 
                and (s.g==o.g).all() 
                and (s.h==o.h).all()):
            return True
        else:
            return False

    def __ne__(s,o):
        return not s.__eq__(o)

    def __add__(s, o):
        if type(o) == type(s):
            # add (+) is defined for two functions, which share the same variables!
            f = s.f + o.f
            g = s.g + o.g
            h = s.h + o.h
        elif np.isscalar(o):
            f = s.f + o
            g = s.g
            h = s.h
        return FGH(f,g,h)

    __radd__ = __add__

    def __sub__(s, o):
        if type(o) == type(s):
            # sub (-) is defined for two functions, which share the same variables!
            f = s.f - o.f
            g = s.g - o.g
            h = s.h - o.h
        elif np.isscalar(o):
            f = s.f - o
            g = s.g
            h = s.h
        return FGH(f,g,h)

    def __neg__(s):
        f = -s.f
        g = -s.g
        h = -s.h
        return FGH(f,g,h)

    def __rsub__(s, o):
        return s.__sub__(o).__neg__()
 
    def __mul__(s, o):
        if type(o) == type(s):
            # mul (*) is defined for two functions, which have different variables!
            # this implies an order for the multiplication and sometimes requires
            # the neutral element I
            f = s.f*o.f
            g = np.append(o.f*s.g, s.f*o.g)
            out = np.outer(s.g,o.g)
            h = np.bmat([[o.f*s.h, out],
                         [out.transpose(), s.f*o.h]])
        elif np.isscalar(o):
            # for multiplication with a scalar
            f = o*s.f
            g = o*s.g
            h = o*s.h
        return FGH(f,g,h)

    __rmul__ = __mul__

    def __matmul__(s, o ):
        if type(o) == type(s):
            # matmul (@) is the multiplication for two functions, which have the same variable
            f = s.f*o.f
            g = o.f*s.g + s.f*o.g
            out = np.outer(s.g,o.g)
            h = o.f*s.h + s.f*o.h + out + out.transpose()
            return FGH(f,g,h)

    def __pow__(s, n):
        if np.isscalar(n):
            f = s.f**n
            g = n*s.f**(n-1)*s.g
            h = n*((n-1)*s.f**(n-2)*np.outer(s.g,s.g)+s.f**(n-1)*s.h)
            return FGH(f,g,h)

    def __truediv__(s, o):
        if type(o) == type(s):
            return s.__mul__(o.__pow__(-1))
        elif np.isscalar(o):
            f = s.f/o
            g = s.g/o
            h = s.h/o
            return FGH(f,g,h)

    def __rtruediv__(s,o):
        if np.isscalar(o):
            return s.__pow__(-1).__mul__(o)

    def __floordiv__(s, o):
        if type(o) == type(s):
            return s.__matmul__(o.__pow__(-1))

    def __abs__(s):
        return s.__pow__(2).__pow__(0.5)

    def sqrt(s):
        return s.__pow__(0.5)

    def exp(s):
        f = np.exp(s.f)
        g = f*s.g
        h = f*(np.outer(s.g,s.g)+s.h)
        return FGH(f,g,h)

    def log(s):
        if abs(s.f) > 0:
            f = np.log(s.f)
            g = s.g/s.f
            h = (s.f*s.h - np.outer(s.g,s.g))/s.f**2
        else:
            f = -np.inf
            g = np.full(s.g.shape, np.nan)
            h = np.full(s.h.shape, np.nan)
        return FGH(f,g,h)

    def __str__(s):
        return f'''value:
{s.f}

gradient:
{s.g.__str__()}

hessian:
{s.h.__str__()}'''

    def __float__(s):
        return s.f

    def __format__(s, str_):
        return s.f.__format__(str_)

    def gradient_norm(s):
        f = np.linalg.norm(s.g)
        g = s.h@s.g/f
        h = np.full(s.h.shape, np.nan)
        return FHG(f,g,h)

    def denanify(s):
        f = s.f
        h = s.h
        g = s.g
        for i, e in enumerate(s.g):
            if np.isnan(e):
                g[i] = 0.0
                h[i,:] = 0.0
                h[:,i] = 0.0
                h[i,i] = 1.0
        return FGH(f,g,h)

def norm(R):
    n = len(R)
    f = np.linalg.norm(R)
    if f > 0.0:
        g = R/f
        h = (np.eye(n)-np.outer(g, g))/f
    else:
        g = np.full(n, np.nan)
        h = np.full((n,n), np.nan)
    return FGH(f,g,h)

def det(A):
    n = np.size(A,0)
    det = 0.0
    x = np.arange(n)
    for i, p in enumerate(permutations(x)):
        # the signature of the permutations is +, -, -, +, +, -, -, +, +, ...
        det += (-1)**((i+1)//2) * np.prod([A[i,j] for (i,j) in zip(x,p)])
    return det

def I(n):
    return FGH(1.0, np.zeros(n), np.zeros((n,n)))

