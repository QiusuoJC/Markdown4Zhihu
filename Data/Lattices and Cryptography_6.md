# 格约化算法

[TOC]

## Lattice reduction algorithms

> 这一节对应教材的 6.12.

我们已经看过了几种密码系统，其安全性依赖于求解各种类型格中的 apprSVP 和/或 apprCVP 问题的困难性。这一节我们描述一种称为 LLL 的算法，它可以在 $C^n$ 的因子范围内解决这些问题其中 $C$ 是一个小常数，$n$ 是格的维数。因此在低维情况下，LLL 算法接近解决 SVP 和 CVP，但在高维时的表现就不太好了。最终，基于格的密码系统的安全性取决于 LLL 和其他格约化算法无法有效地在 $O(\sqrt{n})$​​​​ 的因子范围内解决 apprSVP 和 apprCVP。我们首先介绍高斯的格约化算法，它可以快速解决二维格的 SVP。接下来，我们描述并分析 LLL 算法。之后我们将解释如何将 LLL 与 Babai 算法相结合来解决 apprCVP。最后，我们简要描述 LLL 的一些推广。

### Gaussian lattice reduction in dimension 2

在二维格中寻找最优基的算法主要归功于 Gauss。其基本思想是交替地从一个基向量中减去另一个基向量的倍数，直到无法进一步改进为止。

设 $L\in \R^2$ 是 2 维 lattice，基向量为 $v_1,v_2$，必要的话，交换 $v_1,v_2$ 我们可以假设 $\left\lVert v_1\right\rVert<\left\lVert v_2\right\rVert$。我们现在尝试通过减去 $v_1$ 的倍数来让 $v_2$ 更小一些。如果我们能够减去 $v_1$ 的**任意倍**，那么我们可以将 $v_2$ 替换为向量：
$$
v_2^{*}=v_2-\frac{v_1\cdot v_2}{\left\lVert v_1\right\rVert^2}v_1,
$$
$v_2^{*}$ 是与 $v_1$ 正交的。向量 $v_2^{∗}$ 是 $v_2$ 在 $v_1$ 的正交补(orthogonal complement)上的投影(projection)。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\16.png" style="zoom:67%;"/>
      <figcaption>图 1. 投影图示，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>
注意，向量 $v^{∗}_2$ 不太可能在格 $L$ 中。实际上，我们只允许从 $v_2$ 中减去 $v_1$ 的**整数倍**。所以我们用以下向量替换 $v_2$。
$$
v_2-mv_1\quad \text{with}\quad m=\left\lfloor\frac{v_1\cdot v_2}{\left\lVert v_1\right\rVert^2}\right\rceil.
$$

> 符号 $\lfloor x\rceil$ 表示对 $x$ 四舍五入到整数。

如果 $v_2$ 仍然要比 $v_1$ 长，则停止。否则，我们交换 $v_1,v_2$ 并重复以上过程。Gauss 证明了以上过程会终止并且得到的结果是 $L$ 的一组非常好的基。下面的命题明确地说明了这一点。

**Proposition (Gaussian Lattice Reduction).** 设 $L\in \R^2$ 是 2 维 lattice，基向量为 $v_1,v_2$。下面的算法会终止并得到 $L$ 的一组好的基。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\17.png" style="zoom:67%;"/>
      <figcaption>图 2. Gauss 格约化算法，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

更准确地说，当算法终止时，$v_1$ 向量是 $L$ 上的最短向量，因此算法也能解决 SVP。另外，$v_1,v_2$ 的角度(angle) $\theta$ 满足 $\left\lvert \cos{\theta}\right\rvert\leq\frac{\left\lVert v_1\right\rVert}{2\left\lVert v_2\right\rVert}$，所以 $\frac{\pi}{3}\leq\theta\leq\frac{2\pi}{3}$。

*Proof.* 这里只证明 $v_1$ 是一个最短非零格向量。我们设算法已经终止并返回了向量 $v_1,v_2$。这意味着 $\left\lVert v_2\right\rVert\geq\left\lVert v_1\right\rVert$ 并且：
$$
\frac{\left\lvert v_1\cdot v_2\right\rvert}{\left\lVert v_1\right\rVert^2}\leq \frac{1}{2}.\tag{1}
$$
上面的条件表明我们无法通过让 $v_2$ 减去 $v_1$ 的整数倍来使 $v_2$ 变得更小。

> 这里根据算法终止条件 $m=0$ 即可得到。$m=0$ 说明 $v_1\cdot v_2/\left\lVert v_1\right\rVert^2$ 在四舍五入后为 0，因此分式的取值范围为：$[-\frac{1}{2},\frac{1}{2})$。注意 $-\frac{1}{2}$ 四舍五入后为 0。

我们设 $v\in L$ 是 $L$ 中任意的非零向量，将其写为：
$$
v=a_1v_1+a_2v_2\quad\text{with }a_1,a_2 \in \Z
$$
可以发现：
$$
\begin{align}
\left\lVert v\right\rVert^2&=\left\lVert a_1v_1+a_2v_2\right\rVert^2\\
&=a_1^2\left\lVert v_1\right\rVert^2+2a_1a_2(v_1\cdot v_2)+a_2^2\left\lVert v_2\right\rVert^2\\
&\geq a_1^2\left\lVert v_1\right\rVert^2-2\lvert a_1a_2\rvert(v_1\cdot v_2)+a_2^2\left\lVert v_2\right\rVert^2\\
&\geq a_1^2\left\lVert v_1\right\rVert^2-\lvert a_1a_2\rvert\left\lVert v_1\right\rVert^2+a_2^2\left\lVert v_2\right\rVert^2 &\ \text{from }(1)\\
&\geq a_1^2\left\lVert v_1\right\rVert^2-\lvert a_1a_2\rvert\left\lVert v_1\right\rVert^2+a_2^2\left\lVert v_1\right\rVert^2 &\ \text{since }\left\lVert v_2\right\rVert\geq\left\lVert v_1\right\rVert\\
&=(a_1^2-\lvert a_1\rvert\lvert a_2\rvert+a_2^2)\left\lVert v_1\right\rVert.
\end{align}
$$
对任意的实数 $t_1,t_2$，
$$
t_1^2-t_1t_2+t_2^2=\left(t_1-\frac{1}2t_2\right)^2+\frac{3}4t_2^2=\frac{3}4t_1^2+\left(\frac{1}{2}t_1-t_2\right)^2
$$
等于 0 当且仅当 $t_1=t_2=0$。所以由于 $a_1,a_2$ 均为整数且不会同时为0，那么 $\left\lVert v\right\rVert^2\geq\left\lVert v_1\right\rVert^2$。这就证明了 $v_1$ 是 $L$ 中最小的非零向量。

*Example.* 令 $L$ 的两个基向量为：
$$
v_1 =(66586820,65354729)\quad\text{and}\quad v_2 = (6513996,6393464).
$$
我们首先计算 $\left\lVert v_1\right\rVert^2\approx8.71\cdot 10^{15},\left\lVert v_2\right\rVert^2\approx8.33\cdot 10^{13}$。由于 $v_2$ 更短，于是我们交换它们，即 $v_1 = (6513996,6393464)\quad\text{and}\quad v_2 =(66586820,65354729) .$

下面我们用 $v_2$ 减去 $v_1$ 的一个倍数：
$$
m=\left\lfloor\frac{v_1\cdot v_2}{\left\lVert v_1\right\rVert^2}\right\rceil=\left\lfloor 10.2221\right\rceil=10,
$$
所以我们将 $v_2$ 替换为：
$$
 v_2 −mv_1 =(1446860,1420089).
$$
新向量的 Euclidean 范数为 $\left\lVert v_2\right\rVert^2\approx 4.11\cdot10^{12}$，要比 $\left\lVert v_1\right\rVert^2\approx8.33\cdot 10^{13}$ 更小，所以我们再次交换：
$$
v_1 =(1446860,1420089)\quad\text{and}\quad v_2 = (6513996,6393464).
$$
我们重复上面的过程，$m=\left\lfloor v_1\cdot v_2/\left\lVert v_1\right\rVert^2\right\rceil=\left\lfloor 4.502\right\rceil=5$，又给出了一个新向量：
$$
v_2 −mv_1 =(−720304,−706981),
$$
Euclidean 范数为 $\left\lVert v_2\right\rVert^2\approx 1.01\cdot10^{12}$，所以我们接着交换 $v_1,v_2$。持续这个过程直到算法终止。算法的每一步我们由下面的表格详细列出：

| Step |        $v_1$         |         $v_2$         |  $m$   |
| :--: | :------------------: | :-------------------: | :----: |
|  1   | $ (6513996,6393464)$ | $(66586820,65354729)$ |  $10$  |
|  2   | $(1446860,1420089)$  |  $(6513996,6393464)$  |  $5$   |
|  3   | $(−720304,−706981)$  |  $(1446860,1420089)$  |  $-2$  |
|  4   |    $(6252,6127)$     |  $(−720304,−706981)$  | $-115$ |
|  5   |   $(−1324,−2376)$    |     $(6252,6127)$     |  $-3$  |
|  6   |   $ (2280,−1001)$    |   $ (−1324,−2376)$    |  $0$   |

最后的向量已经足够小了，$(2280,−1001)$ 就是 $L$ 上的 SVP 的一个解。

### The LLL lattice reduction algorithm

Gauss 的格约化算法给出了一个高效的寻找二维格的最短非零向量的方法，然而，随着维数的增加，最短向量问题也变得更加困难。一个重大的进展是1982年 LLL 算法的发表。这一届我们将给出 LLL 算法的完整描述，下一节我们将简要介绍它的一些推广形式。

假设我们有格 $L$ 的一组基。我们的目标是将给定的基转化为“更好的”基。但是什么是"更好"的基呢？我们希望在更好的基中的向量尽可能地短，其开始于我们能找到的最短向量，然后是长度尽可能缓慢增长的向量，直到达到基中的最后一个向量。或者说，我们希望更好基中的向量彼此之间尽可能地正交，即内积 $v_i \cdot v_j$ 尽可能地接近于0。

回忆一下我们之前介绍的 Hadamard 不等式：
$$
\det(L)=\text{Vol}(\mathcal{F})\leq \left\lVert v_1\right\rVert\left\lVert v_2\right\rVert\cdots\left\lVert v_n\right\rVert,
$$
其中 $\text{Vol}(\mathcal{F})$ 是 $L$ 的基本域的体积。基中的向量越接近于正交，不等式也就越接近于等式。

为了帮助我们创造一个更好的基，我们首先创建一个 Gram-Schmidt 正交基。即我们令 $v_1^{*}=v_1$，然后对于 $i \geq 2$ 我们令：
$$
v_i^{*}=v_i-\sum_{j=1}^{i-1}\mu_{i,j}v_j^{*},\quad \text{where}\quad\mu_{i,j}=\frac{v_i \cdot v_j^{*}}{\left\lVert v_j^{*}\right\rVert^2}\quad \text{for}\quad 1\leq j\leq i-1.\tag{2}
$$
向量集合 $\mathcal{B}^{*}=\{v_1^{*},v_2^{*},\dots,v_n^{*}\}$ 是由 $\mathcal{B}=\{v_1,v_2,\dots,v_n\}$ 扩张的向量空间的正交基，但是注意 $\mathcal{B}^{*}$ 并不是 $\mathcal{B}$ 扩张得到的格 $L$ 的基，因为 Gram-Schmidt 变换包含了非整系数的线性组合。然而，我们现在可以证明，这两组基有相同的行列式。

**Proposition.** 设 $\mathcal{B}=\{v_1,v_2,\dots,v_n\}$ 是格 $L$ 的一组基，$\mathcal{B}^{*}=\{v_1^{*},v_2^{*},\dots,v_n^{*}\}$ 是与之相关联的 Gram-Schmidt 正交基，则：
$$
\det(L)=\prod_{i=1}^{n}\left\lVert v_i^{*}\right\rVert.\tag{3}
$$
*Proof.* 将第 $i$ 个向量写作 $v_i=(r_{i1},r_{i2},\dots,r_{in})$。设 $F=F(v_1,\dots,v_n)$ 表示的矩阵的第 $i$ 行为 $v_i$ 的坐标，即：
$$
F=F(v_1,\dots,v_n)=\begin{pmatrix}
{r_{11}}&{r_{12}}&{\cdots}&{r_{1n}}\\
{r_{21}}&{r_{22}}&{\cdots}&{r_{2n}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{r_{n1}}&{r_{n2}}&{\cdots}&{r_{nn}}\\
\end{pmatrix}.
$$

根据之前的命题我们有 $\det(L)=\left\lvert\det F\right\rvert$.

令 $F^{*}=F(v_1^{*},\dots,v_n^{*})$ 是一个与之类似的矩阵(analogous matrix)，只是矩阵的行为行向量 $v_1^{*},\dots,v_n^{*}$。那么 $(2)$ 式告诉我们矩阵 $F$ 和 $F^{*}$ 满足以下关系：
$$
MF^{*}=F,
$$
其中 $M$ 是基变换矩阵(the change of basis matrix)：
$$
M=\begin{pmatrix}
{1}&{0}&{0}&{\cdots}&{0}&{0}\\
{\mu_{2,1}}&{1}&{0}&{\cdots}&{0}&{0}\\
{\mu_{3,1}}&{\mu_{3,2}}&{1}&{\cdots}&{0}&{0}\\
{\vdots}&{\vdots}&{\vdots}&{\ddots}&{\vdots}&{\vdots}\\
{\mu_{n-1,1}}&{\mu_{n-1,2}}&{\mu_{n-1,3}}&{\cdots}&{1}&{0}\\
{\mu_{n,1}}&{\mu_{n,2}}&{\mu_{n,3}}&{\cdots}&{\mu_{n-1,n}}&{1}\\
\end{pmatrix}.
$$
注意 $M$ 是一个下三角矩阵，且对角线元素都是1，所以 $M$ 的行列式$\det(M)=1$。因此：
$$
\det(L)=\left\lvert \det(F)\right\rvert=\left\lvert \det(MF^{*})\right\rvert=\left\lvert \det(M)\det(F^{*})\right\rvert=\left\lvert \det(F^{*})\right\rvert=\prod_{i=1}^{n}\left\lVert v_i^{*}\right\rVert.
$$
这是因为矩阵的行列式可以解释为由其行（或列）向量张成的平行多面体的体积(具体在专栏第2篇文章的 Hadamard 不等式处有解释)。此时由于 $v_i^{*}$ 都是正交的，于是等价于一个 $n$ 维的长方体的体积。

**Definition.** 设 $V$ 是一个向量空间，令 $W\subset V$ 是 $V$ 的一个向量子空间。$W$ (在 $V$ 中)的正交补(orthogonal complement)为：
$$
W^{\bot}=\{v\in V:\ v\cdot w = 0 \quad \text{for all }w\in W\}.
$$
不难看出 $W^{\bot}$ 也是 $V$ 的一个向量子空间，且对于每一个 $v\in V$ 都可以被唯一地写为 $v=w+w^{'}$ 的形式，$w\in W,w^{'}\in W^{\bot}$。

利用正交补的概念，我们可以描述 Gram-Schmidt 构造背后的直观思想如下：
$$
v_i^{*}=\text{Projection of }\ v_i\ \text{onto } \text{Span}(v_1,\dots,v_{i-1})^{\bot}.
$$
尽管 $\mathcal{B}^{*}=\{v_1^{*},v_2^{*},\dots,v_n^{*}\}$ 不是格 $L$ 的一个基，但是我们可以利用集合 $\mathcal{B}^{*}=\{v_1^{*},v_2^{*},\dots,v_n^{*}\}$ 来定义一个在 LLL 算法中非常重要的概念。

**Definition.** 设 $\mathcal{B}=\{v_1,v_2,\dots,v_n\}$ 是格 $L$ 的一组基，$\mathcal{B}^{*}=\{v_1^{*},v_2^{*},\dots,v_n^{*}\}$ 是与之相关联的 Gram-Schmidt 正交基。如果基 $\mathcal{B}$ 满足以下两个条件，则称其为 LLL 约化的。
$$
(\text{Size Condition})\quad \left\lvert \mu_{i,j}\right\rvert=\frac{\left\lvert v_i\cdot v_j^{*}\right\rvert}{\left\lVert v_j^{*}\right\rVert^2}\leq \frac{1}{2}\quad \text{for all }1\leq j<i\leq n.\\
(\text{Lov\'{a}sz Condition})\quad \left\lVert v_i^{*}\right\rVert^2 \geq \left(\frac{3}{4}-\mu_{i,i-1}^2\right)\left\lVert v_{i-1}^{*}\right\rVert^2\quad \text{for all }1< i\leq n.
$$
有很多种方式来表述 Lovász condition，例如其等价于不等式：
$$
\left\lVert v_i^{*}+\mu_{i,i-1}v_{i-1}^{*}\right\rVert^2 \geq \frac{3}{4}\left\lVert v_{i-1}^{*}\right\rVert^2,
$$

也等价于：
$$
\left\lVert \text{Projection of }v_i\text{ onto Span}(v_1,\dots,v_{i-2})^{\bot}\right\rVert \geq \frac{3}{4}\left\lVert \text{Projection of }v_{i-1}\text{ onto Span}(v_1,\dots,v_{i-2})^{\bot}\right\rVert,
$$
因此 Lovász 条件保证了 $\left\lVert v_i^{*}\right\rVert$ 的长度不会比 $\left\lVert v_{i-1}^{*}\right\rVert$ 短太多。

Lenstra、Lenstra 和 Lovász 的基本成果表明，LLL 约化基(reduced basis)是一个好的基，并且在多项式时间内计算出一个 LLL 约化基是可能的。我们首先展示 LLL 约化基具有期望的属性，之后我们将描述 LLL 格约化算法。

**Theorem.** 设 $L$ 是一个 $n$ 维的格。任何 $L$ 的 LLL 约化基 $\{v_1,v_2,\dots,v_n\}$ 有以下两个性质：
$$
\begin{align}
\prod_{i=1}^n \left\lVert v_i\right\rVert &\leq 2^{n(n-1)/4}\det(L),\tag{4}\\
\left\lVert v_j\right\rVert &\leq 2^{(i-1)/2}\left\lVert v_i^{*}\right\rVert\quad \text{for all }1\leq j\leq i\leq n.\tag{5}
\end{align}
$$
此外，一个 LLL 约化基的初始向量满足：
$$
\left\lVert v_1\right\rVert\leq 2^{(n-1)/4}\left\lvert\det(L)\right\rvert^{1/n}\quad \text{and}\quad\left\lVert v_1\right\rVert\leq 2^{(n-1)/2}\min_{0\neq v \in L}\left\lVert v\right\rVert.\tag{6}
$$
因此，LLL 约化基可以在 $2^{(n-1)/2}$ 的因子范围内解决近似最短向量问题(apprSVP)。

*Proof.* 根据 LLL 约化基的定义我们有：
$$
\left\lVert v_j^{*}\right\rVert^2 \geq \left(\frac{3}{4}-\mu_{i,i-1}^2\right)\left\lVert v_{i-1}^{*}\right\rVert^2\geq \frac{1}{2}\left\lVert v_{i-1}^{*}\right\rVert^2.
$$
重复使用上面的式子，我们可以得到有用的估计：
$$
\left\lVert v_{j}^{*}\right\rVert^2\leq 2^{i-j}\left\lVert v_{i}^{*}\right\rVert^2\tag{7}
$$
我们继续计算：
$$
\begin{align}
\left\lVert v_{i}\right\rVert^2&=\left\lVert v_{i}^{*}+\sum_{j=1}^{i-1}\mu_{i,j}v_j^{*}\right\rVert^2\quad&\text{根据 }v_i^{*} 定义,\\
&=\left\lVert v_{i}^{*}\right\rVert^2+\sum_{j=1}^{i-1}\mu_{i,j}^2 \left\lVert v_{j}^{*}\right\rVert^2\quad&\text{since }v_i^{*}\text{ are orthogonal,}\\
&\leq\left\lVert v_{i}^{*}\right\rVert^2+\sum_{j=1}^{i-1}\frac{1}{4}\left\lVert v_{j}^{*}\right\rVert^2\quad&\text{since }\left\lvert\mu_{i,j}\leq \frac{1}{2}\right\rvert,\\
&\leq\left\lVert v_{i}^{*}\right\rVert^2+\sum_{j=1}^{i-1}2^{i-j-2}\left\lVert v_{i}^{*}\right\rVert^2\quad&\text{根据(7)},\\
&=\frac{1+2^{i-1}}{2}\left\lVert v_{i}^{*}\right\rVert^2\\
&\leq 2^{i-1}\left\lVert v_{i}^{*}\right\rVert^2\tag{8}
\end{align}
$$
对于 $1\leq i\leq n$ 两边各自相乘得到：
$$
\prod_{i=1}^n \left\lVert v_{i}\right\rVert^2\leq\prod_{i=1}^n 2^{i-1}\left\lVert v_{i}^{*}\right\rVert^2=2^{n(n-1)/2}\prod_{i=1}^n \left\lVert v_{i}^{*}\right\rVert^2=2^{n(n-1)/2}(\det(L))^2,
$$
两边开平方根即完成 $(4)$ 的证明。

对于任意的 $j\leq i$，我们用 $(7)$ 和 $(8)(i=j)$ 来估计：
$$
\left\lVert v_{j}\right\rVert^2\leq 2^{j-1}\left\lVert v_{j}^{*}\right\rVert^2\leq 2^{j-1}\cdot2^{i-j}\left\lVert v_{i}^{*}\right\rVert^2=2^{i-1}\left\lVert v_{i}^{*}\right\rVert^2.
$$
两边开平方根即完成 $(5)$​ 的证明。

在 $(5)$ 中我们令 $j=1$，对于 $1\leq i \leq n$ 做乘法，并用 $(3)$ 式得到：
$$
\left\lVert v_{1}\right\rVert^n\leq\prod_{i=1}^n 2^{(i-1)/2}\left\lVert v_i^{*}\right\rVert=2^{n(n-1)/4}\prod_{i=1}^n\left\lVert v_i^{*}\right\rVert=2^{n(n-1)/4}\det(L).
$$
两边开 $n$ 次根号得到 $(6)$ 的第一个估计。

为了证明第二个估计，令 $v\in L$ 是一个非零格向量并将其写为：
$$
v=\sum_{j=1}^i a_j v_j =\sum_{j=1}^i b_jv_j^{*}.
$$
其中 $a_i\neq 0$。注意 $a_1,\dots,a_i$ 是整数而 $b_1,\dots,b_i$ 是实数。特别地，$\left\lvert a_i \right\rvert \geq 1$.

根据前面的论述我们知道，对于任意的 $k$，向量 $v_1^{*},\dots,v_k^{*}$ 是相互正交的，同时我们证明了它们所张成的空间与 $v_1,\dots,v_k$ 是同一个。因此：
$$
v\cdot v_i^{*}=a_i v_i\cdot v_i^{*}=b_i v_i^{*}\cdot v_i^{*}\quad \text{and}\quad v_i\cdot v_i^{*}=v_i^{*}\cdot v_i^{*}
$$

> 对 $(2)$ 式两边同时乘 $v_i^{*}$ 即可得到 $v_i\cdot v_i^{*}=v_i^{*}\cdot v_i^{*}$​ 

所以我们可以得到 $a_i=b_i$。于是 $\left\lvert b_i\right\rvert=\left\lvert a_i\right\rvert\geq 1$，再用 $(5)(j=1)$ 式给出估计：
$$
\begin{align}
\left\lVert v\right\rVert^2 &=\sum_{j=1}^i b_j^2\left\lVert v_j^{*}\right\rVert^2\qquad \text{对 }v\text{ 的表达式两边再同乘 }v\\
&\geq b_i^{2}\left\lVert v_j^{*}\right\rVert^2\geq\left\lVert v_j^{*}\right\rVert^2\geq 2^{-(i-1)}\left\lVert v_1\right\rVert^2\\
&\geq 2^{-(n-1)}\left\lVert v_1\right\rVert^2.
\end{align}
$$
两边开平方根即可得到 $(6)$ 的第二个估计。

*Remark.* 在描述 LLL 算法的技术细节之前，我们先简要说明其背后的一般思想。给定一个基 $\{v_1,v_2,\dots,v_n\}$，很容易形成一个满足 Size Condition 的新基。简而言之，我们通过从 $v_k$ 中减去前面向量 $v_1,\dots,v_{k-1}$ 适当的整数倍来使 $v_k$ 变小。在 LLL 算法中，我们分阶段而不是一次性地进行这一操作，我们将看到尺度约化(size reduction)的条件取决于向量的顺序。在进行尺度约化后，我们检查是否满足 Lovász 条件。如果满足，那么我们就拥有了一个（几乎）最优的向量排序。如果不满足，我们则重新排序向量并做进一步的尺度约化。

**Theorem (LLL Algorithm).** 设 $\{v_1,\dots,v_n\}$ 是格 $L$ 的一个基。在图 3 中描述的算法会在有限步内停止并且返回一个格 $L$ 的 LLL 约化基。

更准确地说，令 $B=\max\left\lVert v_i\right\rVert$。则算法执行主循环(Steps[4]–[14])的次数不会超过 $O(n^2\log n+n^2\log B)$。特别地，LLL 算法是一个多项式时间的算法。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\18.png" style="zoom:67%;"/>
      <figcaption>图 3. LLL 格约化算法，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>
*Remark.* LLL 算法的高效实现存在许多挑战。首先，尺度约化和Lovász 条件使用了 Gram-Schmidt 正交化基 $v^{*}_1, \dots, v^{*}_n$ 及其相关的投影因子 $\mu_{i,j} = v_i \cdot v^{*}_j/\left\lVert v^{*}_j\right\rVert^2$。在 LLL 算法的高效实现中，应当按需计算这些量并将其存储起来以供将来使用，仅在必要时重新计算。在图 3 中，我们没有涉及这个问题，因为它与理解 LLL 算法或证明它返回一个 LLL 约化基在多项式时间内是不相关的。

> 教材的 Exercise 6.42，给出了一个更高效的 LLL 算法版本。

另一个主要的挑战源自于，如果尝试使用精确值在整数格上执行 LLL 约化，中间计算会涉及到非常大的数。因此，在处理高维格基时，通常需要使用浮点近似值，这就会引发舍入误差的问题。在这里我们没有足够的空间详细讨论这个实际困难，但读者应该知道这个问题的存在。

> 对于 LLL 的证明有两页多，我后面再来填坑。。。

*Example.* 我们将使用LLL算法来处理一个 6 维格 $L$，其基(有序)由矩阵 $M$ 的行给出：
$$
M=\begin{pmatrix}
19 & 2 & 32 & 46 & 3 & 33\\
15 & 42 & 11 &0 & 3 & 24\\
43 & 15 & 0 & 24 & 4 & 16\\
20 & 44 & 44 & 0 & 18 & 15\\
0 & 48 & 35 & 16 & 31 & 31\\
48 & 33 & 32 & 9 & 1 & 29\\
\end{pmatrix}.
$$
最短向量为 $\left\lVert v_2\right\rVert=51.913$.

LLL 算法的输出是由矩阵 $M^{LLL}$ 的各行组成的基。
$$
M^{LLL}=\begin{pmatrix}
7&−12& −8& 4& 19& 9\\
 −20& 4& −9& 16& 13& 16\\
 5& 2& 33& 0& 15& −9\\
 −6& −7&−20&−21& 8&−12\\
 −10&−24& 21&−15&−6&−11\\
 7& 4& −9&−11& 1& 31\\
\end{pmatrix}.
$$
我们可以检查，这两个矩阵拥有相同的行列式。
$$
\det(M)=\det(M^{LLL})=\pm777406251.
$$
此外，像我们预期的那样，LLL 约化后的矩阵比原矩阵有更好(更大)的 Hadamard 比率，
$$
\mathcal{H}(M)=0.46908\quad \text{and}\quad\mathcal{H}(M^{LLL})=0.88824,
$$
所以约化后的基更加正交。LLL 约化后的基中的最短向量为 $\left\lVert v_1\right\rVert=26.739$，相对于原基底来说是一个显著的改善。这也可以与 Gaussian 期望最短长度(第一节)相比：$\sigma(L)=(3!\det(L))^{1/3}/\sqrt{\pi}=23.062$。 

LLL 算法执行力 19 步交换(swap)操作。从开始到结束 $k$ 的值由以下序列给出：
$$
\begin{align}
2, 2, 3, 2, 3, 4, 3, 2, 2, 3, &4, 5, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6, 5,\\
& 4, 3, 4, 5, 6, 5, 4, 3, 2, 2, 3, 2, 3, 4, 5, 6.
\end{align}
$$
注意，算法几乎完成了两次(它达到了$k=6$)才最终在第三次终止。这说明了随着算法的进行，$k$ 值是如何上下波动的。

我们下面将 $M$ 的行逆序并应用 LLL，此时 LLL 仅执行 11 次交换就给出了基底：
$$
M^{LLL}=\begin{pmatrix}
 −7& 12& 8& −4&−19& −9\\
 20&−4& 9&−16&−13&−16\\
 −28& 11& 12& −9& 17&−14\\
 −6&−7&−20&−21& 8&−12\\
 −7&−4& 9& 11& −1&−31\\
 10& 24&−21& 15& 6& 11\\
\end{pmatrix}.
$$
我们发现两次由相同的最短向量，但是 Hadamard 比率 $\mathcal{H}(M^{LLL})=0.878973$ 略有减小，所以这组基并不与之前一样好。这说明 LLL 算法的输出依赖于基底向量的顺序。

我们再次对原矩阵执行 LLL 算法，这一次我们用 $0.99$ 来代替Lovász Step [8]中的 $\frac 3 4$。此时算法执行力 22 步交换操作，比用 $\frac 3 4$ 时的 19 步交换更多。这并不令人惊讶，因为增大了常数会使得 Lovász 条件更加严格，所以算法更难达到 $k$-增加那一步的条件(即更容易进入 Else 交换那个分支条件)。利用 $0.99$，LLL 算法返回基底：
$$
M^{LLL}=\begin{pmatrix}
 −7& 12& 8& −4&−19& −9\\
 -20&4& -9&16&13&16\\
 6& 7&20&21& -8&12\\
 −28&11& 12& −9& 17&−14\\
 −7&−4& 9& 11& −1&−31\\
 -10& -24&21& -15& -6& -11\\
\end{pmatrix}.
$$
同样，我们得到了相同的最短向量，但是现在基底的 Hadamard 比率为 $\mathcal{H}(M^{LLL})=0.87897$。这比由 $\frac 3 4$ 得到的基底实际上稍微差了点，这也说明了 LLL 算法的输出结果对其参数的依赖性时不可预测的。

### Using LLL to solve apprCVP

在教材的 6.6 节(专栏第3篇文章的 Babai 算法)中讲过，如果格 $L$ 有一个正交基底，那么解决 SVP 和 CVP 都是非常简单的。LLL 算法虽然不能返回一个正交基底，但是它返回了一个准正交(quasi-orthogonal)的基底，即它们彼此接近正交。因此我们能够结合 LLL 算法和 Babai's 算法来构造一个新的算法解决 apprCVP。

**Theorem (LLL apprCVP Algorithm).** 存在一个常数 $C$ ，满足对于给定一组基 $v_1,\dots,v_n$ 下的任意 $n$ 维格 $L$，以下算法可以在 $C^n$ 的因子范围内解决 apprCVP 问题。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\20.png" style="zoom:67%;"/>
      <figcaption>图 4. LLL apprCVP 算法，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

*Proof.* 证明见教材 Exercise 6.44。

*Remark.* 在 [文献](https://link.springer.com/article/10.1007/BF02579403) 中，Babai 提出了两种使用 LLL 算法作为 apprCVP 算法一部分的方法。第一种方法使用我们在 6.6 节描述的最近顶点算法(the closest vertex algorithm)。第二种方法使用最近平面算法(the closest plane)。结合最近平面方法和 LLL 约简基通常比使用最近顶点方法能得到更好的结果。详情请参见教材 Exercise 6.45。

### Generalizations of LLL

有很多 LLL 算法的改进和推广。大部分方法通过增加运行时间来换取更好的输出效果。我们简要描述其中的两种改进，以便让读者了解它们的工作原理和所涉及的权衡(trade-offs)。

LLL 的第一个变体被称作是深插法(deep insertion method)。在标准的 LLL 算法中，交换操作要交换 $v_k$ 和 $v_{k-1}$，这通常允许对新的 $v_k$ 做进一步的尺度约化。而在深插法中，我们将 $v_k$ 插入到 $v_{i-1}$ 和 $v_i$ 之间，其中 $i$ 的选择是为了允许大量的尺度约化。在最坏的情况下，由此产生的算法可能无法在多项式时间内终止，但在实践中，当在大多数格上运行时，带有深度插入的 LLL 运行速度相当快，并且通常返回比基础的 LLL 好得多的基。

LLL 的第二个变体基于 Korkin–Zolotarev 约化基的概念。对于任意的向量序列 $v_1,v_2,\dots$ 和 $i \geq 1$，令 $v_1^{*},v_2^{*},\dots$ 表示相应的 Gram-Schmidt 正交化向量。定义映射：
$$
\pi:L\longrightarrow \R^n,\quad \pi_i(v)=v-\sum_{j=1}^i \frac{v\cdot v_j^{*}}{\left\lVert v_j^{*}\right\rVert^2} v_j^{*}.
$$
我们也定义 $\pi_0$ 为恒等映射，$\pi_0(v)=v$。从几何上说，我们可以将 $\pi_i$ 视为为投影映射(projection map)：
$$
\pi_i: L \longrightarrow \text{Span}(v_1,\dots,v_i)^{\bot}\subset \R^n
$$
从 $L$ 到由 $v_1,\dots,v_i$ 扩张成的正交补空间。

**Definition.** 设 $L$ 是一个格。$L$ 的一个基 $v_1,\dots,v_n$ 被称为是 Korkin-Zolotarev (KZ) 约化的，如果它满足下面的三个条件：

1. $v_1$ 是 $L$ 的最短非零向量
2. 对于 $i=2,3,\dots,n$，选择向量 $v_i$，使得 $\pi_{i-1}(v_i)$ 是 $\pi_{i-1}(L)$​ 中最短的非零向量。
3. 对于 $1\leq i<j \leq n$，我们有 $\left\lvert\pi_{i-1}(v_i)\cdot \pi_{i-1}(v_j)\right\rvert \leq\frac{1}{2}\left\lVert\pi_{i-1}(v_i) \right\rVert^2$.

一个 KZ 约化基通常要比 LLL 约化基更好。特别的，KZ 约化基的第一个向量总是 SVP 的一个解。这并不令人惊讶，目前已知最快的寻找 KZ 约化基的方法，其所需时间随维度呈指数增长。

LLL 算法的分块 KZ (block Korkin–Zolotarev)变体，缩写为 BKZ-LLL，用一个分块约化步骤(block reduction step)取代了标准 LLL 算法中的交换操作。一种看待 LLL 中“交换和尺度约化”步骤的方法是由 $v_{k-1}$ 和 $v_k$ 扩张成的二维格上进行的高斯格约化。在 BKZ-LLL 算法中，我们不是单独处理一个向量，而是处理一个长度为 $\beta$ 的向量块，例如：
$$
(v_k, v_{k+1}, ..., v_{k+\beta-1}),
$$
我们用一个生成相同子格的 KZ 约化基来替换这个块中的向量。如果 $\beta$ 很大，一个明显的缺点是计算 KZ 约化基需要更长的时间。补偿这个额外时间的是算法的最终输出在理论和实践上都得到了改善。

**Theorem.** 如果 BKZ-LLL 算法在一个 $n$ 维格 $L$ 上以 $\beta$ 为分块大小运行，那么算法可以保证在不超过 $O(\beta^{c\beta}n^d)$ 步后停止，其中 $c,d$ 都是非常小的常数。此外，由算法计算得到的最短向量 $v_1$ 满足：
$$
\left\lVert v_1 \right\rVert\leq \left(\frac{\beta}{\pi e}\right)^{\frac{n-1}{\beta-1}}\min_{0\neq v \in L}\left\lVert v\right\rVert.
$$
*Remark.* 上面定理告诉我们 BKZ-LLL 在大约 $\beta^{n/\beta}$ 的因子范围内解决 apprSVP。与之相比，标准的 LLL 算法在大约 $2^{n/2}$ 的因子范围内解决 apprSVP。随着 $\beta$ 的增大，BKZ-LLL 算法的准确度也在增加，而代价是运行时间也随之增加。然而，如果我们想要在 $O(n^\delta)$ 内解决apprSVP，在某个固定的指数 $\delta$ 和大的维数 $n$ 的条件下，那么我们需要取 $\beta \approx n/\delta$，这样 BKZ-LLL 算法的运行时间就会变成 $n$ 的指数级。虽然这些只是最坏情况下的运行时间估计，但实验证据也表明，使用 BKZ-LLL 将 apprSVP 在 $O(n^{\delta})$ 的范围内解决，需要一个与 $n$ 呈线性增长的块大小，因此其运行时间也会随 $n$ 呈指数增长。















