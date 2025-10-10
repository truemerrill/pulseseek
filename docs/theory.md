# Theory

`pulseseek` uses a Lie algebraic representation to solve certain optimization
problems in quantum control theory.  Unlike matrix based representations,
most of the numerical code in this library works on a vector representation
of the Lie algebra.

We strongly encourage users to formally study [Lie theory](https://ocw.mit.edu/courses/18-755-introduction-to-lie-groups-fall-2004/)
to develop a solid understanding of the underlying mathematics.  This short
exposure summarizes some of the key theoretical results that `pulseseek`
uses in it's implementation.

## Lie theory

Let \( \mathfrak{g} \) denote a **Lie algebra**, and let  \( \mathfrak{G} = \exp(\mathfrak{g}) \)
denote its corresponding **Lie group**.  By definition, a Lie algebra 
\( \mathfrak{g} \) is a vector space that is also closed under a bilinear 
operation called the **Lie bracket**, denoted by \([\,\cdot\,,\,\cdot\,]\). Formally,

\[
[x, y] \in \mathfrak{g}
\quad \text{for all} \quad x, y \in \mathfrak{g}.
\]

The Lie bracket is bilinear and satisfies the **Jacobi identity**, two
properties that endow \( \mathfrak{g} \) with a rich algebraic structure
closely related to smooth transformations in \( \mathfrak{G} \).  We further assume that
\( \mathfrak{g} \) is equipped with an **inner product**
\( \langle \cdot , \cdot \rangle \) and the corresponding **norm** 
\( || \cdot || \). These provide the algebra with a geometric structure, 
allowing projection, orthogonality, and numerical stability in computations.

The **exponential map**

\[
\exp : \mathfrak{g} \rightarrow \mathfrak{G}
\]

provides a local correspondence between the algebra and its Lie group.  Through
this map, any group element \( X \in \mathfrak{G} \) near the identity can be expressed as
\( X = \exp(x) \) for some \( x \in \mathfrak{g} \).  The vector \( x \) thus
serves as a **local coordinate** for the group element \( X \), allowing 
computations on \( \mathfrak{G} \) to be carried out in the (typically simpler) vector
space \( \mathfrak{g} \).

## Algebra structure

Let \(E = \{ e_1, e_2, \cdots, e_m \} \) be a basis for the algebra
\( \mathfrak{g} \).  Because both the Lie bracket and inner product are
bilinear, their complete behavior is determined by their action on the basis
elements.

The **structure constants** of the algebra are the coefficients
\( F_{ijk} \in \mathbb{R} \) (or \( \mathbb{C} \)) defined by the expansion

\[
[e_i, e_j] = \sum_{k=1}^{m} F_{ijk} \, e_k.
\]

The structure constants form a rank-3 antisymmetric tensor.

Similarly the **Gram matrix** contains coefficients \( G_{ij} \in \mathbb{R} \)
(or \( \mathbb{C} \)) defined by

\[
\langle e_i, e_j \rangle = G_{ij}
\]

The Gram matrix is a Hermitian matrix that encodes the metric structure of 
\( \mathfrak{g} \).  When the basis is orthonormal, \( G_{ij} = \delta_{ij} \), 
and the inner product simplifies to the usual Euclidean form.  For 
non-orthonormal bases, \( G_{ij} \) allows the computation of inner products,
norms, and projections.

### Numeric representation

In `pulseseek`, the Lie algebra is represented numerically by the tuple
\( (G, F) \), where \( G \) is the Gram matrix defining the inner product and
\( F \) is the structure constant tensor defining the Lie bracket. 
All computations involving inner products or commutators are performed using
these tensors.

This representation enables a simple **vector-based** implementation of Lie
algebra elements, while remaining fully consistent with their corresponding
**matrix representations** in cases where \( \mathfrak{g} \) is a matrix Lie
algebra.

## BCH formulas

The **Baker-Campbell-Hausdorff** (BCH) map is the function
\(Z : \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g} \), satisfying

\[
\exp\!\big(Z(x,y)\big) = \exp(x)\,\exp(y), \qquad x,y \in \mathfrak{g}.
\]

The map \( Z \) admits a formal expansion in the noncommuting variables
\( x \) and \( y \):

\[
Z(x,y) = \sum_{k=1}^{\infty} Z_k(x,y),
\]

where each \(Z_k : \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}\) is a
homogeneous Lie polynomial of total degree \(k\), that is, a \(k\)-linear
combination of nested Lie brackets containing exactly \(k\) occurrences
of \(x\) and \(y\) in total.  Each homogeneous term $Z_k$ can be decomposed
further according to the number of occurrences of \(x\) and \(y\):

\[
Z_k(x,y) = \sum_{\substack{p,q \ge 0 \\ p+q = k}} Z_{(p,q)}(x,y),
\]

where \(Z_{(p,q)}\) denotes the component of bidegree \((p,q)\), formed by all
monomials with \(p\) copies of \(x\) and \(q\) copies of \(y\).  

Explicit formulas for \(Z_k\) are [well known](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula#Explicit_forms),
but at high order these quickly become algebraically complex and difficult to
work with.  `pulseseek` stores pre-computed algebraic terms for \(Z_{(p,q)}\)
up to order \(p + q \leq 15 \).  At runtime, these exact algebraic expressions
are compiled using `jax.jit` into efficient and automatically differentiable
machine code that can be executed on GPUs and other hardware.

### Table of \(Z_{(p,q)}\)

| Total degree | Bidegree \((p, q)\) | \(Z_{(p,q)}(x, y)\) |
|:-------------:|:-------------------:|:-------------------------------|
| 1 | (1, 0) | \( x \) |
|   | (0, 1) | \( y \) |
| 2 | (1, 1) | \( \tfrac{1}{2} [x, y]\) |
| 3 | (2, 1) | \( \tfrac{1}{12} [x,[x,y]]\) |
|   | (1, 2) | \( \tfrac{1}{12} [y,[y,x]]\) |
| 4 | (2, 2) | \( -\tfrac{1}{24}[y,[x,[x,y]]]\) |
| 5 | (4, 1) | \( -\tfrac{1}{720}[x,[x,[x,[x,y]]]]\) |
|   | (3, 2) | \( \tfrac{1}{180}[x, [x, [[x, y], y]]] + \tfrac{1}{360}[[x, [x,y]], [x,y]] \) |
|   | (2, 3) | \( \tfrac{1}{180}[x,[[[x,y], y], y]] + \tfrac{1}{120}[[x,y], [[x,y],y]] \) |
|   | (1, 4) | \( -\tfrac{1}{720}[[[[x,y],y],y],y] \) |


## Multilinear lifting

Although \(Z_{(p,q)}(x,y)\) depends on only two arguments, it is **multilinear**
in the \(p\) appearances of \(x\) and the \(q\) appearances of \(y\) within each
monomial of bidegree \((p,q)\). To make this structure explicit, we introduce a
corresponding multilinear map

\[
\mathcal{F}_{(p,q)} :
\underbrace{\mathfrak{g} \times \cdots \times \mathfrak{g}}_{p\ \text{copies}}
\times
\underbrace{\mathfrak{g} \times \cdots \times \mathfrak{g}}_{q\ \text{copies}}
\longrightarrow \mathfrak{g},
\]

such that the symmetric specialization

\[
Z_{(p,q)}(x,y) = \mathcal{F}_{(p,q)}\!\left( \underbrace{x, x, \dots, x}_{p\ \text{copies}};\, \underbrace{y, y, \dots, y}_{q\ \text{copies}} \right)
\]

recovers the original bidegree-\((p,q)\) term.

The map \(\mathcal{F}_{(p,q)}\) is multilinear in each argument and obeys the
scaling law

\[
\mathcal{F}_{(p,q)}(a x_1, \ldots, a x_p;\, b y_1, \ldots, b y_q)
=
a^{p} b^{q}\,
\mathcal{F}_{(p,q)}(x_1, \ldots, x_p;\, y_1, \ldots, y_q).
\]

for any two scalars \(a\) and \(b\). This “lifted” representation of
\(Z_{(p,q)}\) makes it possible to evaluate each BCH component on distinct
inputs for its \(x\)- and \(y\)-slots.

## Lie polynomials

A **Lie polynomial** with coefficients in \(\mathfrak{g}\) is a formal power series whose
coefficients are elements of \(\mathfrak{g}\):

\[
y(t) = \sum_{n \ge 1} t^{n}\, y_n,
\]

where each \(y_n \in \mathfrak{g}\) and \(t\) is a formal commuting parameter.
Such a series represents a formal curve in the Lie algebra, with coefficients
\(\{y_n\}\) describing its successive orders of variation.

### BCH formulas

Let \(x(t)\) and \(y(t)\) be Lie polynomials with coefficients in \(\mathfrak{g}\):

\[
x(t) = \sum_{r \ge 1} t^{r}\, x_r, \qquad
y(t) = \sum_{s \ge 1} t^{s}\, y_s,
\]

where \(x_r, y_s \in \mathfrak{g}\). We define their BCH composition by

\[
w(t) = Z\!\big(x(t),\, y(t)\big) = \log\!\big(\exp(x(t))\, \exp(y(t))\big),
\]

where \(Z\) is the BCH map defined above.  The function \(w(t)\) is itself a Lie
polynomial.  We show this by substituting the BCH map with its multilinear lifting

\[
w(t) = \sum_{p,q \ge 0} \mathcal{F}_{(p,q)}\!\left(
\underbrace{x(t),\ldots,x(t)}_{p\ \text{copies}};\,
\underbrace{y(t),\ldots,y(t)}_{q\ \text{copies}}
\right).
\]

Then by repeatedly substituting the power series for \(x(t)\) and \(y(t)\) and
grouping terms by powers of \(t\), the result takes the polynomial form 
\(w(t) = \sum_{n \geq 1} t^n w_n\) where the coefficients are

\[
w_n = \sum_{\substack{p,q \ge 0 \\ r_1+\cdots+r_p+s_1+\cdots+s_q = n}} \mathcal{F}_{(p,q)}(x_{r_1},\ldots,x_{r_p}; y_{s_1},\ldots,y_{s_q}).
\]

### Table of \(w_n\)

| \(n\) | \(w_n\) | 
|:--:|:--|
| **1** | \( w_1 = x_1 + y_1 \) | Linear order; 
| **2** | \( w_2 = x_2 + y_2 + \tfrac{1}{2}[x_1, y_1] \) | 
| **3** | \( w_3 = x_3 + y_3 + \tfrac{1}{2}([x_1, y_2] + [x_2, y_1]) + \tfrac{1}{12}\big([x_1,[x_1,y_1]] + [y_1,[y_1,x_1]]\big) \) |
| **4** | \( \begin{aligned} w_4 &= x_4 + y_4 + \tfrac{1}{2}([x_1, y_3] + [x_2, y_2] + [x_3, y_1]) \\ &\quad + \tfrac{1}{12}\big([x_1,[x_1,y_2]] + [x_1,[x_2,y_1]] + [x_2,[x_1,y_1]] \\ &\qquad\quad + [y_1,[y_1,x_2]] + [y_1,[y_2,x_1]] + [y_2,[y_1,x_1]]\big) \\ &\quad - \tfrac{1}{24}[y_1,[x_1,[x_1,y_1]]] \end{aligned} \) |



### Pulse sequences

This BCH construction provides a systematic way to compute Lie group products of
the form \( e^{x(t)} e^{y(t)} = e^{w(t)} \) while remaining entirely within a
polynomial representation of the Lie algebra.  

One of the primary goals of `pulseseek` is to develop an optimization framework
capable of discovering **nontrivial sequences** of Lie algebra elements
\( x_1, x_2, \ldots, x_L \) such that

\[
e^{w(t)} = e^{t x_L} \cdots e^{t x_2} e^{t x_1}
= \prod_{\ell = 1}^{L} \exp(t x_\ell),
\]

and where \( w_n = 0 \) for all \( n \leq N \).  We refer to such a
construction as an **\(\mathcal{O}(N)\) compensating pulse sequence**
of length \( L \).
