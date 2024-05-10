# 格上的最短向量问题、Mingkowski 定理和 Babai 算法

[TOC]

## Short vectors in lattices

> 这一节对应教材的6.5.

格中最基本的计算困难问题为寻找格中最短的非零向量和给定一个不在格中的向量，寻找一个与之最近的向量。这一节我们主要从理论上来讨论这些问题。

> 不过在讨论最短向量问题之前，首先应该证明格中最短向量的存在性。**格的离散性保证了长度最短的非零向量的存在性**[1]。至于再严格的证明我也真的没有找到:sob:
>
> 格的离散性(discret)是指：每个点 $x\in\Z^n$ 在 $\R^n$ 中都存在一个邻域，在该邻域内 $x$​ 是唯一的格点。[2]

### The shortest vector problem and the closest vector problem

我们首先对两个格上的基本问题进行讨论。

**最短向量问题(The Shortest Vector Problem, SVP):** 寻找一个非零格向量 $v$，使得 Euclidean norm $\left\lVert v\right\rVert$ 最小。 

**最近向量问题(The Closest Vector Problem, CVP):** 给定一个不在 $L$ 中的向量 $w\in \mathbb{R}^m$，寻找一个格中的向量 $v\in L$，使得 Euclidean norm $\left\lVert w-v\right\rVert$ 最小。

*Remark.* 注意，格中的最短非零向量不止一个，例如在 $\mathbb{Z}^2$ 中，$(0,\pm1), (\pm1,0)$ 都是 SVP 的解。CVP同理。

SVP 和 CVP 属于计算上的困难问题。此外，即使是 SVP 和 CVP 的近似解，在纯数学或者应用数学的不同领域也有很多有趣的应用。在计算复杂性框架下，SVP 在随机归约假设(randomized reduction hypothesis*)下是 $\mathcal{NP}$-hard，而精确的 CVP 是 $\mathcal{NP}$-hard. 

> *This hypothesis means that the class of polynomial-time algorithms is enlarged to include those that are not deterministic, but will, with high probability, terminate in poly nomial time with a correct result. (教材小注)

在实际中，CVP 被认为比 SVP 稍难一点(a little bit harder)，因为 CVP 经常能够被归约到稍微更高维度(a slightly higher dimension)的 SVP。下面我们介绍 SVP 和 CVP 的几个重要的变体。

**最短基问题(Shortest Basis Problem, SBP):** 寻找格的一个基底 $v_1,\dots,v_n$ 使得在某种意义下是最短的。例如，我们可能会要求：
$$
\max_{1\leq i \leq n} \left\lVert v_i\right\rVert\quad \text{or}\quad \sum_{i=1}^n \left\lVert v_i\right\rVert^2
$$
最小。因此 SBP 有很多不同的版本，这取决于我们如何衡量一组基的“大小(size)”。

**近似最短向量问题(Approximate Shortest Vector Problem, apprSVP):** 令 $\psi(n) \geq 1$ 是 $n$ 的一个函数。在 $n$ 维 lattice $L$ 中，寻找一个非零向量，它的长度不会超过最短非零向量长度的 $\psi(n)$ 倍。即，令 $v_{\text{shortest}}$ 表示 $L$ 中的最短非零向量，寻找非零向量 $v\in L$ 满足：
$$
\left\lVert v\right\rVert \leq \psi(n) \left\lVert v_{\text{shortest}}\right\rVert.
$$
对 $\psi(n)$ 的不同选择会得到不同的 apprSVP。例如：
$$
\left\lVert v\right\rVert \leq 3\sqrt{n}\left\lVert v_\text{{shortest}}\right\rVert \quad \text{or} \quad \left\lVert v\right\rVert \leq 2^{n/2} \left\lVert v_\text{{shortest}}\right\rVert
$$
显然能够解决前一个问题的算法，也能被用来解决后一个问题。

**近似最近向量问题(Approximate Closest Vector Problem,  apprCVP):** 令 $\psi(n) \geq 1$ 是 $n$ 的一个函数。给定一个向量 $w\in \mathbb{R}^m$，寻找一个非零格向量 $v\in L$ 使得：
$$
\left\lVert v-w\right\rVert \leq \psi(n)\cdot \text{dist}(w, L).
$$
其中，$\text{dist}(w,L)$ 表示 $w$ 与格 $L$ 中最近的向量之间的 Euclidean 距离。

### Hermite’s theorem and Minkowski’s theorem

格中最短非零向量的长度，在一定程度上取决于维数(dimension)和行列式(determinant)。下面的定理给出了我们一个显式的上界。

**Theorem (Hermite's Theorem).** 每一个维数为 $n$ 的格均包含了一个非零向量 $v\in L$ 满足：
$$
\left\lVert v\right\rVert \leq \sqrt{n} \det(L) ^{1/n}.
$$
*Remark.* 对于给定的维数 $n$，Hermite 常数(Hermite's constant) $\gamma_n$ 表示使得任何 $n$ 维的格 $L$，都包含一个满足下式的非零向量 $v\in L$ 
$$
\left\lVert v\right\rVert^2 \leq \gamma_n \det(L) ^{2/n}.
$$
的最小值。

根据定义我们可以得到 $\gamma_n \leq n$. 而对于 $\gamma_n$ 的准确值我们只知道 $1\leq n \leq 8$ 和 $n=24$ 的情况：
$$
\gamma_2^2=\frac{4}{3},\quad \gamma_3^3=2,\quad\gamma_4^4=4,\quad,\gamma_5^5=8,\quad\gamma_6^6=\frac{64}{3},\quad\gamma_7^7=64,\quad\gamma_8^8=256
$$
以及 $\gamma_{24}=4$.

在密码学中，我们主要对 $\gamma_n$ 和 $n$ 比较大的时候感兴趣。当 $n$ 很大时， Hermite 常数满足：
$$
\frac{n}{2\pi e} \leq \gamma_n \leq \frac{n}{\pi e},
$$
其中 $\pi=3.14159\dots$，$e=2.71828\dots$ 都是一般的常数。

*Remark.* Hermite 定理有很多版本，能够处理不止一个向量的情形。例如，我们可以证明一个 $n$ 维的 lattice $L$ 总有一组基满足：
$$
\left\lVert v_1\right\rVert\left\lVert v_2\right\rVert\dots\left\lVert v_n\right\rVert\leq n^{n/2}(\det(L))
$$

> 这个证明我能想到的就是利用 Minkowski 第二定理来直接得到，Minkowski 定理将在后面介绍。

结合 Hadamard 不等式：$\left\lVert v_1\right\rVert\left\lVert v_2\right\rVert\dots\left\lVert v_n\right\rVert \geq \det(L)$，可以得到：
$$
\frac{1}{\sqrt{n}} \leq \left( \frac{\det(L)}{\left\lVert v_1\right\rVert\left\lVert v_2\right\rVert\dots\left\lVert v_n\right\rVert}\right)^{1/n} \leq 1
$$
定义基底 $\mathcal{B}=\{v_1,\dots,v_n\}$ 的 Hadamard 比率(Hadamard ratio)为：
$$
\mathcal{H}(\mathcal{B})=\left( \frac{\det(L)}{\left\lVert v_1\right\rVert\left\lVert v_2\right\rVert\dots\left\lVert v_n\right\rVert}\right)^{1/n}
$$
因此 $0<\mathcal{H}(\mathcal{B})\leq 1$，且比值越接近于1，基底向量越趋向于正交(其实就是 Hadamard 不等式的结论)。Hadamard 比率的倒数有时被称为正交性缺陷(orthogonality defect)。

Hermite 定理的证明利用了 Minkowski 定理，我们将在后面介绍。为了介绍 Minkowski 定理，我们引入一个有用的符号表示法并给出一些基本定义。

**Definition.** 对于任意的 $a \in \mathbb{R}^n$ 和任意的 $R >0$，以 $a$ 为中心，半径为 $R$ 的(闭)球(closed ball)是集合：
$$
\mathbb{B}_R(a)=\{x\in \mathbb{R}^n:\ \left\lVert x-a\right\rVert\leq R\}.
$$
**Definition.** 令 $S$ 是 $\mathbb{R}^n$ 的一个子集。

1. $S$ 是有界的(bounded)，如果 $S$ 中的向量长度是有界的。即如果存在一个半径 $R$，使得 $S$ 被包含在球 $\mathbb{B}_R(0)$中，则称 $S$ 有界。
2. $S$​ 是对称的(symmetric)，如果对于 $S$​ 中的每个点 $a$​，则其逆 $-a$​ 也在 $S$​ 中。
3. $S$ 是凸的(convex)，如果 $S$ 中的任意两点 $a$ 和 $b$ 满足，连接 $a$ 和 $b$ 的整个线段完全位于 $S$ 内。
4. $S$ 是闭的(closed)，如果对于点 $a\in \mathbb{R}^n$ 满足，每个以 $a$ 为中心、半径为 $R$ 的球 $\mathbb{B}_R(a)$ 都包含 $S$ 中的点，则 $a$ 属于 $S$​。

> 可以在2维或3维上举个例子帮助理解。

**Theorem (Minkowski's Theorem).** 设 $L \subset \mathbb{R}^n$ 是一个 $n$ 维的 lattice，令 $S \subset \mathbb{R}^n$ 是一个对称凸集合(symmetric convex set)，其体积(volume)满足：
$$
\text{Vol}(S) > 2^n\det(L).
$$
则 $S$ 包含一个非零的格向量。如果 $S$ 同时也是闭的，则条件可以放宽到 $\text{Vol}(S) \geq 2^n\det(L).$​

> 定理也称被为 Minkowski 凸体定理(Minkowski convex body theorem)。

*Proof.* 令 $\mathcal{F}$ 表示 $L$ 的基本域。我们在基本域一节已经讨论过，对于任意 $a\in S$，都能被唯一的表示为：
$$
a=v_a+w_a\quad \text{with}\ v_a\in L\ \text{and}\ w_a \in \mathcal{F}.
$$
我们令 $S^{'} = \frac{1}{2}S$，即将 $S$ 缩小两倍：
$$
S^{'}=\frac{1}{2}S=\left\{\frac{1}{2}a:\ a\in S\right\}
$$
考虑映射：
$$
\begin{align}
f:\ &S^{'} \rightarrow \mathcal{F},\\
&\frac{1}{2}a \mapsto w_{\frac{1}{2}a}.
\end{align}
$$
将 $S$ 缩小 2 倍会使其体积变为原来的 $2^n$，于是：
$$
\text{Vol}(\frac{1}{2}S)=\frac{1}{2^n}\text{Vol}(S) > \det(L) = \text{Vol}(\mathcal{F}).
$$
这里我们利用了假设 $\text{Vol}(S)>2^n \det(L)$。

映射 $f$ 由有限个平移映射([translation map](https://proofwiki.org/wiki/Definition:Translation_Mapping))组成(这里使用了 $S$ 有界的假设)，因此该映射是保持体积的。定义域 $S^{'}$ 的体积严格大于值域 $\mathcal{F}$ 的体积，意味着存在不同的点 $\frac{1}{2}a_1$ 和 $\frac{1}{2}a_2$，映射到了 $\mathcal{F}$​ 中相同的像：
$$
\frac{1}{2}a_1=v_1+w\quad \text{and} \quad \frac{1}{2}a_2=v_2 + w \quad\text{with}\ v_1,v_2\in L\ \text{and}\ w \in \mathcal{F}.
$$

> $S, S^{'}, \mathcal{F}$ 都是 $\mathbb{R}^n$ 的子集合，那么映射 $f$ 相当于是将在 $S^{'}$ 中的那些点/向量给*平移* 到了 $\mathcal{F}$ 中，本质其实都是在 $\mathbb{R}^n$ 中移动点。因此经过有限次的平移后，这些点所构成的体积应该是不变的，即保持体积。
>
> 但现在经过映射后，反而体积变小了，这说明 $f$ 不是一个单射，否则应该有
> $$
> f(S^{'}) \subset \mathcal F \Rightarrow \text{Vol}(S^{'})=\text{Vol}(f(S^{'})) \leq \text{Vol}(\mathcal{F})
> $$
> <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\6.png" style="zoom:50%;" />
>
> 但这就与我们的假设 $\text{Vol}(S)>2^n \det(L)$ 矛盾了，因此有 $S^{'}$ 中不同的原像映射到了相同的 $\mathcal{F}$ 中的像。
>
> <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\7.png" style="zoom:50%;" />

两式相减得到非零向量：
$$
\frac{1}{2}a_1-\frac{1}{2}a_2=v_1-v_2 \in L.
$$
同时：
$$
\underbrace{\frac{1}{2}a_1+\overbrace{(-\frac{1}{2}a_2)}}^{S\text{ is sysmetric, }\text{so } -a_2 \text{ is in } S
}_{
\begin{align}
\text{this is the midpoint of the line}\\
\text{ segment from }a_1\ \text{to}\ -a_2,\ \ \ \\ \text{so it is in}\ S\ \text{by convexity}\ \ \
\end{align}}\in S
$$
因此：
$$
0 \neq v_1 -v_2 \in S\cap L,
$$
于是我们在 $S$ 中构建了一个非零格点。

*Proof of Hermite's theorem.* 设 $L \subset \mathbb{R}^n$ 是一个 lattice，令 $S$ 表示 $\mathbb{R}^n$ 上的以0为中心，边长长为 $2B$ 的一个超方形 ([hypercube](https://zh.wikipedia.org/wiki/%E8%B6%85%E6%96%B9%E5%BD%A2))，
$$
S=\{(x_1,\dots,x_n\in \mathbb{R}^n:\ -B \leq x_i\leq B \quad \text{for all }1 \leq i \leq n\}.
$$
集合 $S$ 满足对称性、封闭性且有界，其体积为：
$$
\text{Vol}(S) = (2B)^n
$$
因此，如果我们令 $B=\det(L)^{1/n}$，则 $\text{Vol}(S)=2^n\det(L)$，此时应用 Minkowski 定理便能推导出存在一个非零向量 $0\neq a \in S \cap L$。用坐标表示 $a=(a_1,\dots,a_n),\ -B \leq a_i\leq B$，根据 $S$ 的定义我们有：
$$
\left\lVert a \right\rVert = \sqrt{a_1^2+\dots+a_n^2} \leq \sqrt{n}B = \sqrt{n} \det(L)^{1/n}.
$$
这就完成了 Hermite 定理的证明。

> 这里补充一下格的逐次最小长度(successive minima) 与 Minkowski 第一第二定理。
>
> 参考论文：格的计算和密码学应用，并将符号与本书做了下统一。
>
> **Definition.** 令 $L$ 是 $n$ 维格，对于 $i\in\{1,\dots,n\}$，我们定义第 $i$ 个逐次最小长度 $\lambda_i(L)$ 为包含 $i$ 个线性无关的格向量的以原点为球心的球的最小半径，即
> $$
> \lambda_i(L)=\min\{r>0:\dim(\text{Span}(L\cap\mathbb{B}_r(0)))\geq i\},
> $$
> 特别地，$\lambda_1(L)$ 是格 $L$ 中最短非零向量的长度。下面两个结果分别被称为 Minkowski 第一和第二定理.
>
> **Theorem.** 对于任意 $n$ 维格 $L$，有
>
> 1. $\lambda_1(L) < \sqrt{n}\det(L)^{\frac{1}{n}}$；
> 2. $\prod_{i=1}^n\lambda_i(L)^{\frac{1}{n}}<\sqrt{n}\det(L)^{\frac{1}{n}}$.

### The Gaussian heuristic

通过将 Minkowski 定理应用于超球面([hypersphere](https://zh.wikipedia.org/wiki/N%E7%BB%B4%E7%90%83%E9%9D%A2))，而不是超立方体(hypercube),可以改进出现在 Hermite 定理中的常数。为了实现这一点，我们需要知道在 $\mathbb{R}^n$ 中球体的体积。

**Definition.** 对于 $s>0$，伽马函数(gamma function) $\Gamma(s)$ 用积分定义为：
$$
\Gamma(s)=\int_0^{\infty} t^s e^{-t}\frac{dt}{t}.
$$
我们列出一些基本性质。

**Proposition.** 

1. 对于所有的 $s>0$，定义 gamma 函数 $\Gamma(s)$ 的积分是收敛的。

2. $\Gamma(1)=1$ 且 $\Gamma(s+1)=s \Gamma(s)$。这使得我们能够将 $\Gamma(s)$ 扩展到所有 $s \in \mathbb{R}$ 上，对于 $s\neq 0,-1,-2,\dots$.

3. 对于所有的整数 $n\geq 1$，我们有 $\Gamma(n+1)=n!$。因此 gamma 函数即为阶乘函数在实数与复数域上的推广。

4. $\Gamma(\frac{1}{2})=\sqrt{\pi}$​.

5. (Stirling's 公式) 当 $s$ 很大时我们有：
   $$
   \Gamma(1+s)^{1/s}\approx\frac{s}{e}.
   $$
   更精确来说，
   $$
   \ln\Gamma(1+s)=\ln(\frac{s}{e})^s+\frac{1}{2}\ln(2\pi s) +O(1)\ \text{as } s \rightarrow \infty.
   $$

$n$ 维空间中的球体体积公式包含了 gamma 函数。

**Theorem.** 令 $\mathbb{B}_R(a)$ 表示 $\mathbb{R}^n$ 中半径为 $R$ 的球体。则其体积为：
$$
\text{Vol}(\mathbb{B}_R(a))=\frac{\pi^{n/2}R^n}{\Gamma(1+\frac{n}{2})}.
$$
当 $n$ 很大时，$\mathbb{B}_R(a)$ 的体积可以近似表示为：
$$
\text{Vol}(\mathbb{B}_R(a))^{1/n}\approx\sqrt{\frac{2\pi e}{n}}R.
$$
*Remark.* 利用上面的定理我们可以改进 Hermite 定理当 $n$ 很大时的情况。球体 $\mathbb{B}_R(0)$ 是有界的、封闭的、凸的且对称的，于是根据 Minkowski 定理，如果我们选择 $R$ 满足：
$$
\text{Vol}(\mathbb{B}_R(0)) \geq 2^n \det(L),
$$
则球体 $\mathbb{B}_R(0)$ 包含了一个非零格点。当 $n$ 很大时，利用球体体积1的近似公式，我们需要选择 $R$ 满足：
$$
\sqrt{\frac{2\pi e}{n}}R \gtrapprox 2\det(L)^{1/n}
$$
根据 $\mathbb{B}_R(0)$ 的定义，有：
$$
\mathbb{B}_R(0)=\{x \in \mathbb{R}^n:\ \left\lVert x\right\lVert \leq R\}
$$
因此，存在一个非零向量 $v \in L$​ 满足：
$$
\left\lVert v \right\rVert\lessapprox\sqrt{\frac{2n}{\pi e}} \cdot (\det(L))^{1/n}
$$
这便通过一个因子 $\sqrt{2/\pi e} \approx 0.484$ 改进了 Hermite 定理。

尽管最短向量的准确界在 $n$ 很大时是未知的，但我们可以基于以下原理的概率论证来估计其范围：

>令 $\mathbb{B}_R(0)$ 是以 0 为中心的大球体。则 $\mathbb{B}_R(0)$ 内的格点数约等于 $\mathbb{B}_R(0)$ 的体积除以基本域 $\mathcal{F}$ 的体积。

这是合理的，因为 $\#(\mathbb{B}_R(0)\cap L)$ 应该近似于 $\mathbb{B}_R(0)$ 中能够容纳的 $\mathcal{F}$ 的数量。

例如，如果我们令 $L=\mathbb{Z}^2$，则这条原理告诉我们一个圆的面积约等于落在该圆内的整数点的个数。

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\8.png" style="zoom:67%;"/>
        <figcaption>图 1. 半径为 2 的圆(笔者所绘)</figcaption></figure>
        </p>

而关于误差项的估计：
$$
\#\{(x,y) \in \mathbb{Z}^2:\ x^2+y^2 \leq R^2\} = \pi R^2 +\text{(error term)}
$$
是一个著名的经典问题。随着维数的增大，问题会更加困难，因为当半径不够大时，由靠近球边界的格点所造成的误差会相当大。因此下面的估计：
$$
\#\{v \in L:\ \left\lVert v \right\rVert \leq R\}\approx \frac{\text{Vol}(\mathbb{B}_R(0))}{\text{Vol}(\mathcal{F})}
$$
在 $n$ 很大且 $R$ 不够大的情况下是有问题的。尽管如此，我们仍然可以寻找使右边等于1的 $R$ 的值，因为从某种意义上说，这个 $R$ 值是我们可能首次在球内发现非零格点的那个半径值。

考虑 $n$ 很大的情形，我们用球体体积的估计值公式计算。令：
$$
\sqrt{\frac{2\pi e}{n}}R \approx\text{Vol}(\mathbb{B}_R(0))^{1/n}\quad \text{equal to}\quad \text{Vol}(\mathcal{F})=\det(L),
$$
解得：
$$
R\approx \sqrt{\frac{n}{2\pi e}}(\det(L))^{1/n}.
$$
我们便推出了下面的启发式算法。

**Definition.** 设 $L$ 是 $n$ 维的 lattice。高斯的期望最短长度(Gaussian expected shortest length)是：
$$
\sigma(L) = \sqrt{\frac{n}{2\pi e}}(\det(L))^{1/n}.
$$
高斯的启发式(Gaussian heuristic)方法指的是：一个随机选择的格中，最短非零向量将满足：
$$
\left\lVert v_{\text{shortest}}\right\rVert \approx \sigma(L).
$$
更精确来说，若 $\epsilon > 0$ 固定，那么对于所有足够大的 $n$，一个随机选择的 $n$ 维格满足：
$$
(1-\epsilon) \sigma(L) \leq\left\lVert v_{\text{shortest}}\right\rVert\leq(1+\epsilon)\sigma(L).
$$
*Remark.* 对于较小的 $n$ 值，使用体积的精确公式更好，此时高斯的期望最短长度为：
$$
\sigma(L)=\frac{\Gamma(1+n/2)}{\sqrt{\pi}}(\det(L))^{1/n}
$$
我们会发现高斯启发式方法在量化格中 SVP 的困难程度时很有用。特别是，如果一个特定格 $L$ 的实际最短向量明显比 $\sigma(L)$​ 短，那么诸如 LLL 等格约化算法在定位最短向量时似乎就会容易得多。

*Example.* 设 $(m_1,\dots,m_n,S)$ 是一个背包问题。相关联的格 $L_{M,S}$ 是由 $(*)$ 矩阵的行生成的。矩阵 $L_{M,S}$ 的维度为 $n+1$，行列式为 $\det(L_{M,S})=2^nS$。在子集和问题一节中说过，$S$ 的大小满足 $S=O(2^{2n})$，所以 $S^{1/n}\approx 4$。所以我们可以估计高斯的最短长度为：
$$
\begin{align}
\sigma(L_{M,S})&=\sqrt{\frac{n+1}{2\pi e}}(\det(L_{M,S}))^{1/(n+1)}=\sqrt{\frac{n+1}{2\pi e}}(2^nS)^{1/(n+1)}\\
&\approx\sqrt{\frac{n}{2\pi e}}\cdot2S^{1/(n+1)}\approx\sqrt{\frac{n}{2\pi e}}\cdot8\approx 1.936\sqrt{n}.\\
\end{align}
$$
这就证明了在子集和一节所说的，格 $L_{M,S}$ 包含一个长度为 $\sqrt{n}$ 的向量 $t$，并且知道 $t$ 就可以求得子集和问题的解。因此，解决格 $L_{M,S}$ 的 SVP 问题就很可能解决子集和问题。有关使用格中的方法解决子集和问题的进一步讨论，可以见专栏最后一篇文章的第二个例子。

我们会发现高斯启发式方法在量化寻找格中短向量的难度方面很有用。特别是，如果特定格 $L$ 的实际最短向量明显短于 $\sigma(L)$，那么像 LLL 这样的格约化算法在定位最短向量时似乎要容易得多。

将高斯启发式方法应用于 CVP 也有类似的结果。设 $L \subset \mathbb{R}^n$ 是一个 $n$ 维的 lattice，$w\in \mathbb{R}^n$ 是一个随机点，那么我们期望与 $w$ 最接近的格向量满足：
$$
\left\lVert v - w\right\rVert \approx\sigma(L).
$$
与 SVP 类似，如果 $L$ 包含一个与 $w$ 之间的距离比 $\sigma(L)$​ 小得多的向量，则格约化算法在解决 CVP 时就会更容易。

## Babai’s algorithm and using a “good” basis to solve apprCVP

> 这一节对应教材的6.6.

如果格 $L \subset \mathbb{R}^n$ 有一组相互正交的基 $v_1,\dots,v_n$，即满足：
$$
v_i \cdot v_j = 0 \quad\text{for all }i\neq j,
$$
则我们可以轻松解决 SVP 和CVP。为解决 SVP，我们观察到 $L$ 中的任何向量的长度都由下面公式给出：
$$
\left\lVert a_1v_1+\dots+a_nv_n\right\rVert^2=a_1^2\left\lVert v_1 \right\rVert^2+\dots+a_n^2\left\lVert v_n \right\rVert^2.
$$
因为 $a_1,\dots,a_n\in \mathbb{Z}$，所以 $L$ 中的最短非零向量就是集合 $\{\pm v_1,\dots,\pm v_n\}$ 中的最短向量。即 
$$
v_{\text{shortest}}=\{v_i:\ \left\lVert v_i \right\rVert=\min\{\left\lVert v_1 \right\rVert, \dots, \left\lVert v_n \right\rVert\} \}
$$
类似地可以解决 CVP。我们想要寻找 $L$ 中的一个最短向量，使其与给定向量 $w\in \mathbb{R}^n$ 的距离最近。我们首先将 $w$ 表示为：
$$
w=t_1v_1+\dots+t_nv_n \quad \text{with }t_1,\dots,t_n\in \mathbb{R}.
$$
那么对于 $v=a_1v_1+\dots+a_nv_n \in L$，我们有：
$$
\left\lVert v-w \right\rVert^2=(a_1-t_1)^2\left\lVert v_1 \right\rVert^2+\dots+(a_n-t_n)^2\left\lVert v_n \right\rVert^2.
$$
$a_i$ 是整数，因此上式要想取得最小值，我们只需将每个 $a_i$ 设置为与相应的 $t_i$ 最为接近的整数即可。

如果基中的向量是相互正交的，那么我们很有可能能够成功解决 CVP；但是如果基向量高度不正交，那么该算法就不会运行得很好。我们简要讨论一下潜在的几何原理，然后描述一般的方法，最后以一个二维的例子作结。

$L$ 的一组基确定了一个基本域 $\mathcal{F}$，我们在基本域一节已经证明了：用 $L$ 中的元素对 $\mathcal{F}$ 进行平移将会得到整个 $\mathbb{R}^n$ 空间，因此任何 $w\in \mathbb{R}^n$ 都有 $\mathcal{F}$ 的唯一一个平移 $\mathcal{F}+v,\ v\in L$。我们将平行六面体(parallelepiped) $L+v$ 中最靠近 $w$ 的顶点(vertex)作为我们对 CVP 的假设解。找到最近的顶点其实是很容易，因为：
$$
w=v+\epsilon_1v_1+\epsilon_2v_2+\dots+\epsilon_nv_n\quad \text{for some }0\leq \epsilon_1,\epsilon_2,\dots,\epsilon_n <1,
$$
则我们只需对 $\epsilon_i$ 进行如下替换：
$$
\epsilon_i = \left\{
\begin{aligned}
0,&\quad \text{if }\ \epsilon_i<\frac{1}{2}\\
1,&\quad \text{if }\ \epsilon_i\geq\frac{1}{2}\\
\end{aligned}
\right.
$$
下图展示了整个过程：

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\9.png" style="zoom:67%;"/>
<figcaption>图 2. 尝试用给定的基本域来求解 CVP，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>

观察图 2，看上去这个过程一定有效，但这是因为图中的基向量彼此相对较为正交(reasonably orthogonal to one another)。图 3 说明了同一个格内的两组不同的基。第一个基是“好的(good)”，因为这些向量相当正交(fairly orthogonal)；第二个基是“坏的(bad)”，因为基向量之间的角度非常小。

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\10.png" style="zoom:67%;"/>
<figcaption>图 3. 同一个格的两组不同的基，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>

如果我们尝试使用一个坏的基来求解 CVP，就像图 4 所示，我们可能会遇到问题。非格点的目标点实际上非常接近一个格点(见图 4 中的 Target Point 与 Closest Lattice Point)，但由于平行四边形过于细长，最靠近目标点的顶点实际上相当远(Target Point 与 Closest Vertex)。需要注意的是，随着格的维度增加，这些困难会变得更加严重。在二维或三维，甚至四维或五维中可视化的例子，并不能充分展示在基不够正交的情况下，最近顶点算法在解决 CVP 甚至是 apprCVP 上的失败程度。

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\11.png" style="zoom:67%;"/>
<figcaption>图 4. 对于“坏的”基，Babai's algorithm 的效果会很差。源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>

**Theorem (Babai’s Closest Vertex Algorithm).** 设 $L \subset \mathbb{R}^n$ 是一个 $n$ 维的 lattice，$v_1,\dots,v_n$ 是其一组基，令 $w$ 是 $\mathbb{R}^n$​ 中任意的一个向量。如果基底中的向量相互足够正交，那么我们有以下算法来解决 CVP。

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\12.png" style="zoom:67%;"/>
<figcaption>图 5. Babai's algorithm。源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>

一般来说，如果基中的向量彼此较为正交，那么该算法可以解决某种形式的 apprCVP。但是，如果基向量高度非正交，那么算法返回的向量通常与 $w$​ 相距甚远。

*Example.* 设 $L \subset \mathbb{R}^2$ 是一个 2 维的 lattice。我们给定一组基：
$$
v_1=(137,312) \quad \text{and}\quad v_2=(215,-187).
$$
我们将用 Babai's algorithm 来寻找 $L$ 的一个向量，使其与下面的向量最为接近：
$$
w=(53172, 81743).
$$
首先我们将 $w$ 表示为 $v_1,v_2$ 的实系数线性组合的形式。即我们需要寻找 $t_1, t_2 \in \mathbb{R}$ 满足：
$$
w = t_1 v_1+t_2v_2.
$$
我们可以得到两个线性方程：
$$
53172=137t_1+215t_2 \quad \text{and} \quad 81743=312t_1-187t_2.
$$
或者我们可以用矩阵的形式表示，
$$
(53172,81743)=(t_1,t_2)\left(
\begin{array}
1137 & 312\\
215 &-187\\
\end{array}
\right).
$$
不管哪种方式，我们都能很轻易地计算 $(t_1,t_2)$。最后求出 $t_1\approx296.85,\ t_2\approx 58.15$。Babai's algorithm 告诉我们将 $t_1,\ t_2$ 圆整(round，即四舍五入)到最近的整数，然后计算：
$$
\begin{align}
v&=\lfloor t_1\rceil v_1+\lfloor t_2\rceil
v_2\\
&=297\cdot(137,312)+58\cdot(215,-187)\\
&=(53159,81818).
\end{align}
$$
> $\lfloor t_1\rceil$ 符号表示对 $t_1$ 四舍五入为整数。

$v\in L$ 且  $v$ 应该接近于 $w$。我们发现：
$$
\left\lVert v-w \right\rVert \approx 76.12
$$
的确足够小。这是可以预测的，因为给定基中的向量彼此相当正交，这一点可以从 Hadamard 比率就可以看出：
$$
\mathcal{H}(v_1,v_2)=\left(\frac{\det(L)}{\left\lVert v_1\right\rVert \left\lVert v_2\right\rVert} \right)^{1/2}\approx \left(\frac{92699}{340.75 \times 284.95}\right)^{1/2}\approx 0.977
$$
非常接近于 1.

我们现在尝试用一组新的基来解决同样的问题：
$$
v_1^{'}=(1975,438)=5v_1+6v_2 \quad\text{and}\quad v_2^{'}=(7548,1627)=19v_1+23v_2.
$$
线性方程组
$$
(53172,81743)=(t_1,t_2)\left(
\begin{array}
11975 & 438\\
7548 & 1627\\
\end{array}
\right)
$$
的解为 $(t_1,t_2)\approx (5722.66,-1490.34)$，于是我们令：
$$
v^{'}=5723v_1^{'}-1490v_2^{'}=(56405,82444).
$$
则 $v^{'}\in L$，但 $v^{'}$ 并没有足够接近 $w$，因为：
$$
\left\lVert v^{'}-w \right\rVert \approx 3308.12.
$$
基底 $\{v_1^{'},v_2^{'}\}$ 的非正交性也反映在 Hadamard 比率上：
$$
\mathcal{H}(v_1^{'},v_2^{'})=\left(\frac{\det(L)}{\left\lVert v_1^{'}\right\rVert \left\lVert v_2^{'}\right\rVert} \right)^{1/2}\approx \left(\frac{92699}{2022.99 \times 7721.36}\right)^{1/2}\approx 0.077
$$

## 参考

[1] 格的计算和密码学应用 

[2] A Decade of Lattice Cryptography 的 2.2.1
