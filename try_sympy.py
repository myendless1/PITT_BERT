import sympy
from sympy.abc import x, y, t, z


def nabla(u: sympy.Function):
    symbols = u.free_symbols
    # keep axis symbols
    if t in symbols:
        symbols.remove(t)
    return sum(u.diff(s) for s in symbols)


# 定义分量
def component(u: sympy.Function):
    symbols = u.free_symbols
    # keep axis symbols
    if t in symbols:
        symbols.remove(t)


def nabla_cross(u: sympy.Function):
    symbols = u.free_symbols
    # keep axis symbols
    if t in symbols:
        symbols.remove(t)
    # 2D
    if len(symbols) == 2:
        x = symbols.pop()
        y = symbols.pop()
        return sympy.Matrix([u.diff(x)])


u = sympy.Function('u')
u = u(x, y, t)
ux = u.diff(x)
print(nabla(ux))
v = sympy.Function('v')
v = v(x, y, z)
print(nabla(v))
print(sympy.Matrix([u, ux]))
