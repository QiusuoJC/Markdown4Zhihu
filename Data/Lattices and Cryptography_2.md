# 格的基本定义及性质

[TOC]

## Lattices: Basic definitions and properties

> 这一节对应教材的 6.4.

这一节我们正式开始介绍格的内容。

### 格的定义

**Definition.** 令 $v_1,\dots,v_n \in \mathbb{R}^m$ 是一组线性无关向量。由 $v_1, \dots,v_n$ 生成(generate)的格(lattice) $L$ 是 $v_1,\dots,v_n$ 的线性组合，其中系数取自 $\mathbb{Z}$,
$$
L = \{a_1v_1 + a_2v_2 + \dots + a_nv_n:\ a_1,a_2,\dots,a_n \in \mathbb{Z}\}.
$$
$L$ 的基是任何一组能够生成 $L$ 的线性无关的向量。任何两个基都有相同数量的元素。基中向量的个数称为格 $L$ 的维数(dimension)。

>这里也会看到其他的定义方式，如 Oded Regev 的[课程笔记]([Lattices in Computer Science (Fall 2004) (nyu.edu)](https://cims.nyu.edu/~regev/teaching/lattices_fall_2004/))中，会将 $m$ 称为格的维数(dimension)，而 $n$ 称为格的秩(rank)。当 $n=m$ 时称格为满秩格(full rank lattice)。一般密码学中都是讨论满秩格。
>
>但在本书以及在 [Hermite’s Constant and Lattice Algorithms]([Nguyen_HermiteConstant.pdf (ens.fr)](https://www.di.ens.fr/~pnguyen/Nguyen_HermiteConstant.pdf)) 的 Definition 6 中，都是把基中向量的个数称为维数。我个人倾向于认为 $n$ 应该是维数，因为类比线性空间中维数的定义，维数是基中向量的个数，那么 $L$ 的基中向量个数是 $n$，因此维数应该是 $n$。
>
>$n$ 可以称为秩，这个没问题，因为格也可以用矩阵表示：
>$$
>L=\{Bx \vert x\in \Z^n\}.
>$$
>其中 $B=(v_1,\dots,v_n)^T$ 是基底组成的 $m\times n$ 的矩阵。
>
>根据矩阵秩的定义可以得到 *秩=列秩=行秩=行向量的极大线性无关组=* $n$。
>当然这只是个小问题。。。

设 $v_1,v_2,\dots,v_n$ 是 $V$ 的一个基底，$w_1,w_2,\dots,w_n$ 是 $V$ 中 $n$ 个向量。则 $w_j$ 可以写为基的线性组合：
$$
w_1 = a_{11}v_1 + a_{12}v_2+\dots+a_{1n}v_n,\\
w_2 = a_{21}v_1 + a_{22}v_2+\dots+a_{2n}v_n,\\
\vdots \\
w_2 = a_{n1}v_1 + a_{n2}v_2+\dots+a_{nn}v_n,\\
$$
由于是格，这里的系数都是整数。

我们在线性空间中研究过基变换，我们来看一下格中两个基的关系。分别令 $W, U$ 表示 $w_j$ 和 $v_i$ 两个列向量，$A$ 表示整系数矩阵：
$$
A=\begin{pmatrix}
{a_{11}}&{a_{12}}&{\cdots}&{a_{1n}}\\
{a_{21}}&{a_{22}}&{\cdots}&{a_{2n}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{a_{n1}}&{a_{n2}}&{\cdots}&{a_{nn}}\\
\end{pmatrix}
$$
则有 $W = A\cdot U$. 我们考虑利用 $w_j$ 来表示 $v_i$, 此时只需要对 $A$ 求逆便能得到 $U = A^{-1}\cdot W$. 在格中，线性组合的系数必须都是整数，所以 $A^{-1}$ 中的元素也一定均为整数。注意到：
$$
1 = \det(I) = \det(AA^{-1}) = \det(A)\cdot \det(A^{-1})
$$
而根据行列式的定义，整数矩阵的行列式一定是整数（行列式的定义为某行/列元素与其代数余子式的乘积再求和，只涉及到整数的加法和乘法，所以得到的结果一定是整数），于是 $\det(A), \det(A^{-1})$ 均为整数，从而只能得到$\det(A) = \pm 1$. 这就证明了如下结果：

**Proposition.** 格 $L$ 的任意两个基，其基变换矩阵中俄元素均为整数，且行列式等于 $\pm 1$.

为了计算方便，我们经常会考虑向量坐标取自整数的格。例如：
$$
\mathbb{Z}^n=\{(x_1,x_2,\dots,x_n):\ x_1,x_2,\dots,x_n \in \mathbb{Z}\}
$$
为所有整数坐标的向量所构成的格。我们可以直观看一下 $\mathbb{Z}^2$ 上的格：

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\2.png" style="zoom:67%;"/>
        <figcaption>图 2. 格的一个实例，笔者所绘</figcaption>
        </figure>
</p>



**Definition.** 一个整数格(integral or integer lattice)是指所有整数坐标的向量所构成的格。等价来说，一个整数格是加法群 $\mathbb{Z}^m$​ 的一个子群。

*Remark.* 如果 $L\subset \mathbb{R}^m$ 是一个 $n$ 维的格，则 $L$ 的一个基可以被写为 $n$ 行 $m$ 列的矩阵 $U$, 设 $v_i = (u_{i1}, \dots, u_{im})$ 即有：
$$
U=(v_1, \dots,v_n)^T =\begin{pmatrix}
{u_{11}}&{u_{12}}&{\cdots}&{u_{1m}}\\
{u_{21}}&{u_{22}}&{\cdots}&{u_{2m}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{u_{n1}}&{u_{n2}}&{\cdots}&{u_{nm}}\\
\end{pmatrix}
$$
$L$ 的一个新的基底可以通过左乘一个 $n\times n$ 的矩阵 $A$ 来得到。$A$ 中元素均为整数且行列式为 $\pm1$ (这里其实就是重复了上面的那个命题).

格还有一种更为抽象的定义，其结合了几何与代数的理念。

**Definition.** $\mathbb{R}^m$ 的子集 $L$ 是一个加法子群，如果其对于加法和减法封闭。我们称 $L$ 是一个离散加法子群(discrete additive subgroup) 如果存在一个正数 $\epsilon > 0$, 对于所有的 $v \in L$ 满足如下性质：
$$
L \cap\{w \in \mathbb{R}^m:\ \left\lVert v-w \right\rVert < \epsilon\}=\{v\}.
$$
换句话说，如果在 $L$ 中取任意的一个向量 $v$，并在其周围做一个半径为 $\epsilon$ 的实心球，则球内除 $v$ 之外没有任何其他 $L$ 中的点。

**Theorem.** $\mathbb{R}^m$​ 的一个子集是格当且仅当其是一个离散加法子群。

> 教材没有给出证明，以下证明为个人理解。

*Proof.* 充分性是容易证的，首先格 $L$ 是 $\R^m$ 的加法子群。又因为格中存在最短向量，只需要令 $\epsilon = \text{Shortest vector length}$ 即可。必要性的证明从直观上想，根据离散加法子群的定义 $L$ 一定是由一些离散的点组成的，于是 $L$ 应该是一个格。

### 基本域

**Definition.** 令 $L$ 为 $n$ 维 lattice，$v_1,\dots,v_n$ 是 $L$ 的一个基。$L$ 在这组基下的基本域(fundamental domain/fundamental parallelepiped)是集合：
$$
\mathcal{F}(v_1,\dots,v_n)=\{t_1v_1+t_2v_2+\dots+t_nv_n:\ 0\leq t_i<1 \}
$$
下图展示了一个2维格上的基本域。

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\3.png" style="zoom:67%;"/>
        <figcaption>图 3. 格与基本域，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
</p>



$\mathcal{F}(v_1,\dots,v_n)$ 还可以写成 $\mathbb{R}^n/L$​​. 

>*Proof.*
>$$
>\begin{align}
>\mathbb{R}^n/L &= \{v + L:\ v \in \mathbb{R}^n\}\\
>&= \{(\alpha_1v_1+\alpha_2v_2+\dots+\alpha_nv_n)+L:\  \alpha_i \in \mathbb{R}\}\\
>&=\{(t_1+a_1)v_1+\dots+(t_n+a_n)v_n+L:\ 0\leq t_i<1,\ a_i \in \mathbb{Z}\}\\
>&= \{(t_1v_1+\dots+t_nv_n)+(a_1v_1+\dots+a_nv_n)+L \}\\
>&=\{(t_1v_1+\dots+t_nv_n) + L:\ 0\leq t_i<1\}\\
>&=\mathcal{F}
>\end{align}
>$$

下面的命题说明了基本域在学习格中的重要性。

**Proposition.** 设$L\subset \mathbb{R}^n$ 是 $n$ 维 lattice，令 $\mathcal{F}$ 是 $L$ 的基本域。则每一个向量 $w\in \mathbb{R}^n$ 都可以被写成如下形式：
$$
w = t+v\quad \text{for a unique}\ t \in \mathcal{F}\ \text{and a unique}\ v \in L
$$
等价来说，当 $v$ 遍历格 $L$ 中的向量时，平移后的基本域(the translated fundamental domains)的并集：
$$
\mathcal{F}+v = \{t+v:\ t \in \mathcal{F}\}
$$
恰好覆盖整个 $\mathbb{R}^n$​. 下图展示了经过 $L$ 中的向量平移后的基本域 $\mathcal{F}$ 恰好覆盖了整个 $\mathbb{R}^n$.

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\4.png" style="zoom:67%;"/>
        <figcaption>图 4. 利用格中向量对基本域平移，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
</p>



*Proof.* 令 $v_1,\dots,v_n$ 是 $L$ 的一个基，其生成的基本域为 $\mathcal{F}$. 则 $v_1,\dots,v_n$ 在 $\mathbb{R}^n$ 上线性无关，于是它们也是 $\mathbb{R}^n$ 的一个基。因此，任意的 $w\in \mathbb{R}^n$ 都能被写为形如：
$$
w = \alpha_1v_1+\alpha_2v_2+\dots+\alpha_nv_n \quad \text{for some}\ \alpha_1, \dots, \alpha_n \in \mathbb{R}.
$$
我们将每个 $\alpha_i$ 稍作变形：
$$
\alpha_i = t_i+a_i \quad \text{with}\ 0\leq t_i <1\ \text{and}\ a_i \in \mathbb{Z}
$$
从而将变形后的 $\alpha_i$ 代入原式得到：
$$
w = \overbrace{t_1v_1+t_2v_2+\dots+t_nv_n}^{\text{this is a vector}\ t \in \mathcal{F}}+\overbrace{a_1v_1+a_2v_2+\dots+a_nv_n}^{\text{this is a vector}\ v \in L}
$$

这便证得了 $w$ 可以被表示为我们想要的形式。但证明还没结束，下面我们还需要证明 $t$ 和 $v$ 的唯一性。证明唯一性通用的方法就是假设有两个，最后推出它们是相等的。

我们假设 $w = t+v=t^{'}+v^{'}$ 是其两种表示形式，则有：
$$
\begin{align}
(t_1+a_1)v_1+(t_2+a_2)v_2+\dots+(t_n+a_n)v_n \\
=(t_1^{'}+a_1^{'})v_1+(t_2^{'}+a_2^{'})v_2+\dots+(t_n^{'}+a_n^{'})v_n.
\end{align}
$$
由于 $v_1,\dots,v_n$ 是相互独立的，所以有：
$$
t_i+a_i=t_i^{'}+a_i^{'}\quad \text{for all}\ i = 1,2,\dots,n.
$$
因此
$$
t_i-t_i^{'}=a_i^{'}-a_i \in \mathbb{Z}
$$
是一个整数。但 $t_i$ 和 $t_i^{'}$ 大于等于0且严格小于1，于是要想让 $t_i-t_i^{'}$ 是整数只能是 $t_i=t_i^{'}$， 因此 $t=t^{'}$. 并且：
$$
v=w-t=w-t^{'}=v^{'}
$$
这就完成了上述命题的完整证明。

基本域的体积(volume)是格中重要的一个不变量。

**Definition.** 设$L$ 是 $n$ 维 lattice，令 $\mathcal{F}$ 是 $L$ 的基本域。则 $\mathcal{F}$ 的 $n$ 维体积称为是 $L$ 的行列式(determinant) ，有时也被称为是协体积(covolume). 用 $\det(L)$ 来表示。

>注意到格 $L$ 本身是没有体积的，因为它是一个可数点的集合。如果 $L$ 是包含在 $\mathbb{R}^n$ 中且其维度为 $n$，那么 $L$ 的协体积被定义为商群 $\mathbb{R}^n/L$ 的体积。

如果将基向量 $v_1,\dots,v_n$ 看作是描述基本域(parallelepiped) $\mathcal{F}$ 边长的给定长度的向量，那么对于给定长度的基向量，当这些向量两两正交时，所得到的体积(volume)是最大的。这导致了格的行列式有以下重要的上界：

**Proposition (Hadamard's Inequality).** 令 $L$ 是一个 lattice，取 $L$ 任意的一组基 $v_1,\dots,v_n$, 且 $\mathcal{F}$ 是 $L$ 的一个基本域，则有
$$
\det(L) = \text{Vol}(\mathcal{F}) \leq \left\lVert v_1 \right\rVert \left\lVert v_2 \right\rVert \dots \left\lVert v_n \right\rVert
$$
基底越接近于正交，则 Hadamard 不等式越趋向于等式。即上式右侧部分表示基底正交时求得的体积。

> [行列式](https://zh.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F)可以看作是有向面积或体积的概念在一般的(即更高维)欧几里得空间中的推广。即矩阵的行列式可以解释为由其行（或列）向量张成的平行多面体的（定向的）体积。这里我们可以通过2维和3维的小例子来感受一下这个不等式。
>
> - 2维以平行四边形和矩形为例。
>
>   <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\5.png" style="zoom:50%;" />
>
>   显然，当边相互正交时，面积最大，即同边长情况下，矩形的面积要大于平行四边形的面积，且当平行四边形的边趋近于垂直时，其面积也趋近于等于 $S_1$​.
>
> - 3维以平行六面体和长方体为例。
>
>   考虑体积公式体积等于底面积乘高: $V=S\cdot h$, 在对应棱长相等的情况下，长方体的体积要大于平行六面体的体积。 

如果格 $L \in\mathbb{R}^n$ 中且 $L$ 的维数为 $n$, 那么计算格 L 的行列式就相对容易。下一个命题描述了这个公式，这种情况也是我们最感兴趣的。

**Proposition.** 设 $L \subset\mathbb{R}^n$ 是 $n$ 维 lattice，令 $v_1,\dots,v_n$ 是 $L$ 的一组基，$\mathcal{F} = \mathcal{F}(v_1,\dots,v_n)$ 是相对应的基本域。用坐标表示第 $i$ 个基向量：
$$
v_i = (r_{i1}, r_{i2},\dots, r_{in})
$$
将向量 $v_i$ 的坐标作为矩阵的行向量,
$$
F=F(v_1,\dots,v_n)=\begin{pmatrix}
{r_{11}}&{r_{12}}&{\cdots}&{r_{1n}}\\
{r_{21}}&{r_{22}}&{\cdots}&{r_{2n}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{r_{n1}}&{r_{n2}}&{\cdots}&{r_{nn}}\\
\end{pmatrix}.
$$
则 $\mathcal{F}$ 的体积由下面的公式给出：
$$
\text{Vol}(\mathcal{F}(v_1,\dots,v_n))=\left\lvert \det(F(v_1,\dots,v_n))\right\rvert.
$$

> 对于一般情况，即非满秩格，其体积表示为 $\det{L} = \sqrt{\det(BB^{T})}$.

*Proof.* 需要用到对多变量的积分。

*Example.* 考虑由如下三个线性无关向量生成的3维格 $L\subset \mathbb{R}^3$：
$$
v_1=(2,1,3),\ v_2 = (1,2,0),\ v_3=(2,-3,-5).
$$
则有：
$$
F(v_1,v_2,v_3)=\begin{pmatrix}
2&1&3\\
1&2&0\\
2&-3&-5\\
\end{pmatrix}.
$$
因此，格的体积为：
$$
\det(L) = \left\lvert\det(F) \right\rvert = 36
$$
**Corollary.** 设 $L \subset\mathbb{R}^n$ 是 $n$ 维 lattice，则 $L$ 的每一个基本域都有相同的体积。因此 $\det(L)$ 是格 $L$ 的不变量。

*Proof.* 令 $v_1,\dots,v_n$ 和 $w_1,\dots,w_n$ 分别生成了 $L$ 的两个基本域，并令 $F(v_1,\dots,v_n)$ 和 $F(w_1,\dots,w_n)$ 是与之相关联的矩阵。根据前面的命题，两个基做转换只需对其中一个基左乘一个行列式为 $\pm1$ 的 $n\times n$ 的矩阵 $A$ 即可。
$$
F(v_1,\dots,v_n)=AF(w_1,\dots,w_n)
$$
$v_1,\dots,v_n$ 生成的基本域的体积为：
$$
\begin{align}
\text{Vol}(\mathcal{F}(v_1,\dots,v_n))
&=\left\lvert \det(F(v_1,\dots,v_n)) \right\rvert\\
&=\left\lvert \det(AF(w_1,\dots,w_n)) \right\rvert\\
&=\left\lvert \det(A) \right\rvert \left\lvert \det(F(w_1,\dots,w_n)) \right\rvert\\
&=\left\lvert \det(F(w_1,\dots,w_n)) \right\rvert\\
&=\text{Vol}(\mathcal{F}(w_1,\dots,w_n))
\end{align}
$$

