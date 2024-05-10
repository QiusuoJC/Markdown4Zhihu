# 卷积多项式环、NTRU 公钥密码系统和 NTRU 作为格上的密码系统

[TOC]

## Convolution polynomial rings

> 这一节对应教材的 6.9.

这一节我们将介绍一种用于 NTRU 公钥密码系统的特殊多项式商环(polynomial quotient ring)。

**Definition.** $N$ 是一个正整数。秩(rank)为 $N$ 的卷积多项式环(the ring of convolution polynomial)是商环：
$$
R=\mathbb{Z}[x]/(x^N-1).
$$
类似地，模 $q$ 的卷积多项式环是商环：
$$
\begin{align}
R&=(\mathbb{Z}/q\mathbb{Z})[x]/(x^N-1)\\
&=\mathbb{Z}_q[x]/(x^N-1).
\end{align}
$$
每一个 $R$ 或 $R_q$ 中的元素都可以被唯一表示为形如：
$$
a_0+a_1x+a_2x^2+\dots+a_{N-1}x^{N-1}
$$
其中系数分别取自 $\mathbb{Z}$ 或 $\mathbb{Z}/q\mathbb{Z}$。我们观察到，与更一般的多项式商环相比，在环 $R$ 和 $R_q$ 中进行运算更加容易，因为多项式 $x^N - 1$ 的形式非常简单。关键在于，当我们对 $x^N - 1$ 取模时，我们只是要求 $x^N$ 等于 1。因此，只要 $x^N$ 出现，我们就可以用 1 来代替它。例如，如果我们有一项 $x^k$，则我们可以将 $k$ 写为 $k=iN+j$，其中 $0\leq j<N$ 并且：
$$
x^k=x^{iN+j}=(x^N)^i\cdot  x^j = 1^i \cdot x^j = x^j.
$$
总的来说，$x$ 幂次的指数可以通过模 $N$ 来化简。

多项式：
$$
a(x)=a_0+a_1x+a_2x^2+\dots+a_{N-1}x^{N-1} \in R
$$
可以用其系数向量表示：
$$
(a_0,a_1,a_2,\dots,a_{N-1}) \in \mathbb{Z}^N,
$$
在 $R_q$ 上的多项式同理。多项式加法也与一般的系数向量的加法相对应：
$$
a(x)+b(x) \longleftrightarrow(a_0+b_0,a_1+b_1,a_2+b_2,\dots,a_{N-1}+b_{N-1}).
$$
$R$ 上的多项式乘法就会复杂一些。我们用 $\star$ 表示 $R$ 和 $R_q$ 上的乘法，以区分一般的多项式乘法。

**Proposition.** 两个多项式 $a(x),b(x)\in R$ 的乘积由下面的公式给出：
$$
a(x)\star b(x) =c(x) \quad \text{with} \quad c_k=\sum_{i+j\equiv k\ \pmod N}a_ib_j
$$
定义 $c_k$ 的求和公式是对所有满足条件 $i+j\equiv k\ (\text{mod}\ N)$ 的 $i$ 和 $j$ 进行求和，其中 $i$ 和 $j$ 的取值范围都是从 0 到 $N-1$。对于 $a(x),b(x)\in R_q$ 的多项式的乘法也是由相同的公式给出，只是系数 $c_k$ 还要对 $q$ 取模。

*Proof.* 我们首先计算一般的多项式乘法，之后在利用关系式 $x^N=1$ 来化简。
$$
\begin{align}
a(x)\star b(x)&=\left(\sum_{i=0}^{N-1}a_ix^i\right)\star\left(\sum_{j=0}^{N-1}b_jx^j\right)\\
&=\sum_{k=0}^{2N-2}\left(\sum_{i+j=k}a_ib_j\right)x^k\\
&=\sum_{k=0}^{N-1}\left(\sum_{i+j=k}a_ib_j\right)x^k+\sum_{k=0}^{N-2}\left(\sum_{i+j=k+N}a_ib_j\right)x^k\\
&=\sum_{k=0}^{N-1}\left(\sum_{i+j\equiv k\ \pmod N}a_ib_j\right)x^k.
\end{align}
$$
*Remark.* 两个向量的卷积为：
$$
(a_0,a_1,a_2,\dots,a_{N-1})\star(b_0,b_1,b_2,\dots,b_{N-1})=(c_0,c_1,c_2,\dots,c_{N-1}),
$$
 其中 $c_k$ 由上面的命题给出定义。我们交替使用 $\star$ 来表示环 $R$ 和 $R_q$ 中的卷积以及向量的卷积。

存在一个从 $R$ 到 $R_q$ 的自然映射，我们只需将多项式的系数对 $q$ 取模。这个对 $q$ 取模的映射满足：
$$
\begin{align}
(a(x)+b(x))\ \text{mod}\ q&=(a(x)\ \text{mod}\ q)+(b(x)\ \text{mod}\ q),\\
(a(x)\star b(x))\ \text{mod}\ q&=(a(x)\ \text{mod}\ q)\star (b(x)\ \text{mod}\ q).
\end{align}
$$
映射 $R\rightarrow R_q$ 是一个环同态(ring homomorphism)。

除了从 $R$ 到 $R_q$ 的映射之外，我们通常也需要一种一致的方式来进行反向映射，即从 $R_q$ 映射回 $R$。在众多的提升(lifting)方式中，我们选择下面这种方式：

**Definition.** 令 $a(x)\in R_q$。 从 $a(x)$ 到 $R$ 的中心提升(centered lift)是 $R$ 中唯一的一个多项式 $a^{'}(x)$ 满足：
$$
a^{'}(x)\ \text{mod}\ q=a(x)
$$
其中系数取自区间：
$$
-\frac{q}{2} < a_i^{'} \leq \frac{q}{2}.
$$
例如，如果 $q=2$，则 $a(x)$​ 的中心提升(centered lift)是一个二进制多项式。

> centered lift 不知道如何翻译更加准确

*Remark.* 这种从 $R_q$ 到 $R$ 的提升映射(lifting map)并不保持运算。换句话说，提升后的和或乘积并不一定等于对应和或乘积的提升。即不一定是环同态。

*Example.* $R$ 中的多项式很少有乘法逆元，但是在 $R_q$ 中情况就不一样了。例如，令 $N=5$ 且 $q=2$。则多项式 $1+x+x^4$ 在 $R_2$ 上是由乘法逆元的，因为：
$$
(1+x+x^4)\star(1+x^2+x^3)=1+x+x^2+2x^3+x^4+x^6+x^7=1
$$
$N=5$ 则  $x^6=x$ 且 $x^7=x^2$。当 $q$ 是一个素数时，多项式的扩展 Euclidean 算法告诉我们哪些多项式是可逆元，以及如何在环 $R_q$ 中计算它们的逆元。

**Proposition.** 设 $q$ 是一个素数，则 $a(x)\in R_q$ 存在乘法逆元当且仅当：
$$
\gcd(a(x),x^N-1)=1\quad\text{in }\mathbb{Z}_q[x].
$$
如果上式成立，则 $a(x)^{-1}\in R_q$ 能够用扩展 Euclidean 算法计算，通过寻找多项式 $u(x),v(x)\in \mathbb{Z}_q[x]$ 满足：
$$
a(x)u(x) + (x^N-1)v(x)=1.
$$
则 $a(x)^{-1}=u(x)\in R_q$。

*Example.* 我们以 $N=5$ 和 $q=2$ 为例来给出在 $(1+x+x^4)^{-1}\in R_2$ 的详细计算过程。首先我们用 Euclidean 算法来计算 $1+x+x^4, 1-x^5\in \mathbb{Z}_2[x]$ 的最大公因子(注意因为是模 2，于是有 $1-x^5=1+x^5$)：
$$
\begin{align}
x^5+1&=x\cdot(x^4+x+1)+(x^2+x+1),\\
x^4+x+1&=(x^2+x)(x^2+x+1)+1.\\
\end{align}
$$
所以 $\gcd=1$​。由上面的第二个式子对 1 做替换得到：
$$
\begin{align}
1&=(x^4+x+1)+(x^2+x)(x^2+x+1)\\
&=(x^4+x+1)+(x^2+x)(x^5+1+x(x^4+x+1))\\
&=(x^4+x+1)(x^3+x^2+1)+(x^5+1)(x^2+x)
\end{align}
$$
因此：
$$
(x^4+x+1)^{-1}=1+x^2+x^3 \in R_2.
$$
*Remark.* 无论 $q$ 是否为素数(prime)，环 $R_q$ 都有完美的意义，而且在某些情况下，取 $q$ 为合数(composite)可能是更好的，例如 $q = 2^k$。一般来说，如果 $q$ 是素数 $p$ 的幂次，那么为了计算 $a(x)$ 在 $R_q$ 中的逆元，首先在 $R_p$ 中计算逆元，然后将这个值“提升(lift)”到 $R_{p^2}$ 中的逆元，再提升到 $R_{p^4}$ 中的逆元，以此类推。类似地，如果 $q = q_1q_2 \cdots q_r$，其中每个 $q_i = p_i^{k_i}$ 都是素数幂，我们可以首先在 $R_{q_i}$ 中计算逆元，然后用中国剩余定理将逆元组合起来。

## The NTRU public key cryptosystem

> 这一节对应教材的 6.10.

基于整数分解问题或者离散对数问题的密码系统都是定义在群上的，因为他们底层的困难问题只涉及一种运算。对于 RSA，Diffie-Hellman 和 ElGamal，群是模 $m$ 的单位群，其中模数 $m$ 可能是素数或合数，群运算是模 $m$ 的乘法。对于 ECC，群是模 $p$ 的椭圆曲线上的点集，群运算是椭圆曲线加法。

环是具有两种运算(加法和乘法)的代数结构，这两种运算通过分配律(distributive law)相关联。在本节中，我们描述 NTRU 公钥密码系统。NTRU 自然地使用了卷积多项式环来描述，但其底层的数学困难问题也可以被解释为格中的 SVP 或 CVP。我们将在后面一节讨论其与格的联系。

### The NTRU public key cryptosystem

这一节我们讨论 NTRU(读作 $en-tr\bar{u}$) 公钥密码系统。对于整数 $N>1$ 和两个模数 $p,q$，我们令 $R,R_p,R_q$ 表示卷积多项式环：
$$
R=\mathbb{Z}[x]/(x^N-1),\quad R_p=\mathbb{Z}_p[x]/(x^N-1),\quad R_q=\mathbb{Z}_q[x]/(x^N-1)
$$
我们可以将多项式 $a(x)\in R$ 视为 $R_p$ 或 $R_q$ 中的元素，方法是将其系数模 $p$ 或 $q$ 约减。反过来，我们可以使用中心提升(centered lift)将元素从 $R_p$ 或 $R_q$ 移动到 $R$。我们对参数 $N$、$p$ 和 $q$ 做了很多假设，特别是要求 $N$ 为素数，且 $\gcd(N,q) = \gcd(p,q) = 1$。

> 这些假设的原因在课后练习 6.30 和 6.33 中有解释。简单来说：
>
> 1. 如果 $\gcd(p,q) \neq1$ 即 $p|q$，则 Eve 能够在不知道私钥的情况下恢复出消息；
> 2. 如果 $N$ 不为素数，则 Eve 能够更容易地恢复出私钥

在描述 NTRU 密码系统之前,我们还需要一个符号。

**Definition.** 对于任意的正整数 $d_1,d_2$，我们令：
$$
\mathcal{T}(d_1,d_2)=\left\{
\begin{align}
&a(x)\text{ has }d_1\ \text{coefficients equal to }1,\\
a(x)\in R:\ & a(x)\text{ has }d_2\ \text{coefficients equal to }-1,\\
& a(x)\text{ has all other coefficients equal to }0\\
\end{align}
\right\}.
$$
多项式 $\mathcal{T}(d_1,d_2)$ 被称为三进制多项式(ternary/trinary polynomials)。它们和二进制多项式(binary polynomials)类似，后者的系数只能取 0 或 1。

下面正式开始介绍 NTRU 密码系统。Alice 或一些可信权威(trusted authority)选择公共参数 $(N,p,q,d)$，满足 $N,p$ 都是素数，$\gcd(p,q)=\gcd(N,q)=1$，且 $q>(6d+1)p$。Alice 的私钥包含两个随机选择的多项式 
$$
f(x)\in \mathcal{T}(d+1,d)\quad\text{and}\quad g(x)\in \mathcal{T}(d,d).
$$
随后，Alice 计算逆：
$$
F_q(x)=f(x)^{-1} \in R_q\quad\text{and}\quad F_p(x)=f(x)^{-1}\in R_p.
$$
如果多项式的逆不存在的话，Alice 只需丢弃当前的 $f(x)$ 并重新选择一个新的即可。Alice 在 $\mathcal{T}(d+1,d)$ 上选择 $f(x)$ 而不是 $\mathcal{T}(d,d)$，因为 $\mathcal{T}(d,d)$ 上的元素在 $R_q$ 上不存在逆元。

Alice 接着计算：
$$
h(x)=F_q(x)\star g(x) \in R_q.
$$
多项式 $h(x)$ 作为 Alice 的公钥，私钥为 $(f(x),F_p(x))$。此外，Alice 可以只存储 $f(x)$，在需要的时候再重新计算 $F_p(x)$。

Bob 的明文是多项式 $m(x)\in R$，系数取自 $-\frac 1 2p$ 到 $\frac{1}{2}p$。即，明文 $m$ 是 $R_p$ 中的一个多项式经过中心提升后得到的 $R$ 上的多项式。Bob 选择一个随机多项式(临时密钥) $r(x)\in \mathcal{T}(d,d)$ 并计算：
$$
e(x)\equiv ph(x)\star r(x)+m(x)\pmod q.
$$

密文 $e(x) \in R_q$.

Alice 在收到 Bob 的密文之后，运行解密算法通过计算：
$$
a(x)\equiv f(x)\star e(x)\pmod{q}.
$$
之后，将 $a(x)$ 中心提升到 $R$ 中的一个元素，并模 $p$.
$$
b(x)\equiv F_p(x)\star a(x)\pmod{p}.
$$
若参数选择合理，我们现在验证多项式 $b(x)$ 就等于 $m(x)$。NTRU 公钥密码系统也被称为 $\text{NTRUE}_\text{NCRYPT}$，总结于下图：

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\14.png" style="zoom:67%;"/>
<figcaption>图 1. NTRU 公钥密码系统，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>


**Proposition.** 如果 NTRU 所选参数 $(N,p,q,d)$ 满足：
$$
q>(6d+1)p,
$$
则 Alice 计算得到的多项式 $b(x)$ 就是 Bob 的明文 $m(x)$。

*Proof.*  我们首先要更精确地确定 Alice 计算 $a(x)$ 时的形式。
$$
\begin{align}
a(x)&\equiv f(x)\star e(x)\pmod{q}\\
&\equiv f(x)\star(ph(x)\star r(x)+m(x))\pmod{q}\quad &\text{展开} e(x)\\
&\equiv pf(x)\star F_q(x)\star g(x)\star r(x)+f(x)\star m(x)\pmod{q}\quad &\text{展开} h(x)\\
&\equiv p  g(x)\star r(x)+f(x)\star m(x) \pmod{q} \quad &F_q(x) = f(x)^{-1}\\
\end{align}
$$
考虑多项式 
$$
p  g(x)\star r(x)+f(x)\star m(x),\tag{*}
$$
只在 $R$ 上计算而不模 $q$。我们需要界定最大可能系数。$g(x),r(x)\in \mathcal{T}(d,d)$，若在卷积 $g(x)\star r(x)$ 中，所有的 1 相匹配，所有的 -1 相匹配，此时最大的系数值为 $2d$。类似地，$f(x) \in \mathcal{T}(d+1,d)$，$m(x)$ 的系数在 $-\frac{1}{2}p$ 到 $\frac{1}{2}p$ 之间，因此 $f(x)\star m(x)$ 的最大的系数值为 $(2d+1)\cdot \frac{1}{2}p$。因此,即使 $g(x) \star r(x)$ 的最大系数恰好与 $f(x) \star m(x)$ (这一项在教材里写的是 $r(x) \star m(x)$，个人认为是作者笔误)的最大系数重合，$(*)$ 式的最大系数的绝对值最多为：
$$
p\cdot 2d+(2d+1)\cdot\frac{1}{2}p = \left(3d+\frac{1}2\right)p.
$$
因此我们的假设 $q>(6d+1)p$ 保证了 $(*)$ 式的每个系数都严格小于 $\frac{1}2 q$。因此，当 Alice 在 $R_q$ 中计算 $a(x) \pmod{q}$ 时，就会提升到 $R$ 中，她便能恢复出 $(*)$ 式的准确值。换句话说，
$$
a(x)=p  g(x)\star r(x)+f(x)\star m(x),
$$
在 $R$ 中是成立的，就不需要再模 $q$ 了。

剩下的就简单了。Alice 用 $F_p(x)$ 乘 $a(x)$，将结果模 $p$ 得到：
$$
\begin{align}
b(x)&=F_p(x)\star a(x)\\
&=F_p(x)\star (p  g(x)\star r(x)+f(x)\star m(x))\\
&\equiv F_p(x)\star f(x)\star m(x)\pmod p\\
&\equiv m(x)\pmod p.
\end{align}
$$
*Remark.* $q>(6d+1)p$ 的条件保证了解密不会失败。但是，在证明过程中我们也看到，即使对于小一点的 $q$，解密也是有可能成功的。因为 $g(x)$ 和 $r(x)$ 的正系数和负系数完全排列一致的可能性非常小，$f(x)$ 和 $m(x)$ 也是如此。所以，出于效率和降低公钥大小的考虑，选择一个小一点的 $q$ 或许更有优势。在这种情况下，估计解密失败的概率就是一个精细的问题了。解密失败的概率必须非常小(例如,小于 $2^{-80}$)，因为解密失败有可能向攻击者泄露私钥信息。

*Remark.* 注意到 NTRU 是一个概率密码系统的例子，因为同一个明文 $m(x)$ 会因为临时密钥 $r(x)$ 的不同选择而得到不同的加密结果。于是，无论是用不同的临时密钥加密两个相同的消息，还是用相同的临时密钥加密两个不同的明文，对于 Bob 来说都不是好的做法。标准的做法是用明文的 hash 值来生成临时密钥。

*Remark.* 如果多项式 $f(x)\in\mathcal{T}(d+1,d)$ 的系数较小，那么它的逆 $F_q(x)\in R_q$ 的系数在模 $q$ 的意义下往往是均匀随机的分布。这并不是一个定理，但却是实验观察到的结果。例如，我们令 $N=11，q=73$，选择一个随机的多项式：
$$
f(x)=x^10 +x^8 −x^3 +x^2 −1 \in\mathcal{T}(3,2).
$$
则 $f(x)$ 在 $R_q$ 上是可逆的，它的逆
$$
F_q(x)=22x^{10}+33x^9+15x^8+33x^7−10x^6+36x^5−33x^4−30x^3+12x^2−32x+28
$$
的系数看起来是随机的。类似地，公钥和密文的系数：
$$
h(x)\equiv F_q(x)\star g(x)\pmod{q}\quad\text{and}\quad e(x)\equiv pr(x)\star h(x)+m(x)\pmod{q},
$$
在模 $q$ 的意义下看起来也是随机分布的。

*Remark.* 使用基于格的密码系统的一个动机是它们的执行速度比基于离散对数和因子分解的密码系统更快。那么 NTRU 有多快呢?在加密和解密过程中最耗时的部分是卷积操作。通常情况下，计算卷积 $a\star b$ 需要 $N^2$ 次乘法运算，因为每个系数实质上是两个向量的点积。但是 NTRU 中所需的卷积形式为 $r\star h$，$f\star e$ 和 $F_p\star a$，其中 $r$、$f$ 和 $F_p$ 是三进制多项式。因此这些卷积可以不需要任何乘法运算就能计算出来，只需要大约 $\frac{2}{3}N^2$ 次加法和减法运算。如果 $d$ 小于 $N/3$，前两个卷积只需要 $\frac{2}3dN$ 次加法和减法。因此 NTRU 的加密和解密过程需要 $O(N^2)$ 步，而每步都是非常快速的。

*Example.* 我们给出 NTRU 的一个小例子，公共参数设置为：
$$
 (N,p,q,d)=(7,3,41,2).
$$
我们有：
$$
 41 = q>(6d+1)p =39,
$$

因此解密将会成功。Alice 选择：
$$
f(x)=x^6−x^4+x^3+x^2−1 \in\mathcal{T}(3,2)\quad \text{and}\quad g(x)=x^6+x^4−x^2−x \in\mathcal{T}(2,2).
$$
并计算逆：
$$
\begin{align}
F_q(x)&=f(x)^{−1} \pmod{q} =8x^6 +26x^5 +31x^4 +21x^3 +40x^2 +2x+37\in R_q,\\
F_p(x)&=f(x)^{−1} \pmod{p} = x^6 +2x^5 +x^3 +x^2 +x+1\in R_p.
\end{align}
$$
她存储 $(f(x),F_p(x))$ 作为私钥，计算并发布公钥：
$$
h(x)=F_q(x)\star g(x)=20x^6 +40x^5 +2x^4 +38x^3 +8x^2 +26x+30\in R_q.
$$
Bob 决定发送给 Alice 消息：
$$
m(x)=−x^5 +x^3+x^2−x+1
$$
临时密钥为：
$$
r(x)=x^6 −x^5 +x−1.
$$
Bob 计算并发送给 Alice 密文：
$$
e(x) \equiv pr(x)\star h(x)+m(x)\equiv 31x^6+19x^5+4x^4+2x^3+40x^2+3x+25 \pmod{q}.
$$
解密过程就很简单了。首先她计算：
$$
f(x)\star e(x) \equiv x^6 +10x^5 +33x^4 +40x^3 +40x^2 +x+40 \pmod{q}.
$$
随后获得：
$$
a(x)=x^6 +10x^5 −8x^4 −x^3 −x^2 +x−1\in R
$$
最后，对 $a(x)$ 模 $p$ 并计算：
$$
F_p(x)\star a(x) \equiv 2x^5 +x^3 +x^2 +2x+1 \pmod{p}.
$$
就能得到 Bob 的明文 $m(x)=-x^5+x^3+x^2-x+1.$

### Mathematical problems underlying NTRU

之前提到，公钥 $h(x)$ 的系数在模 $q$ 的意义下看起来是随机的，但这里也存在一个隐藏的关系：
$$
f(x)\star h(x)\equiv g(x)\pmod{q},
$$
其中 $f(x)$ 和 $g(x)$ 的系数都非常小。因此通过寻找私钥来破解 NTRU 就变成了解决下面的问题：

<p align="center">
    <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\15.png" style="zoom:67%;"/>
<figcaption>图 2. NTRU 密钥恢复问题，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
        </p>


*Remark.* NTRU 密钥恢复问题的解并不是唯一的，因为如果 $(f(x),g(x))$ 是一个解的话，那么 $(x^k\star f(x),x^k\star g(x))$ 也是一个解，其中 $0\leq k<N$。多项式 $x^k\star f(x)$ 称为 $f(x)$ 的一个旋转(rotation)，因为其系数被循环移动了 $k$ 个位置。从某种意义上说，旋转充当了私有解密密钥，因为用 $x^k \star f(x)$ 进行解密会得到旋转后的明文 $x^k \star m(x)$。

更一般地，任何有着足够小系数的多项式对 $(f(x),g(x))$，且满足上面的隐藏关系式的都能作为 NTRU 的一个解密密钥。例如，如果 $f(x)$ 是原始的解密密钥且 $\theta(x)$ 的系数非常小，则 $\theta(x)\star f(x)$ 可能也是一个解密密钥。

*Remark.* 为什么人们会认为 NTRU 密钥恢复问题是一个难解的数学问题？首先必须满足的一个要求是，该问题无法通过暴力破解(brute-force)或碰撞搜索(collision search)在实际中解决。我们将在本节后面讨论此类搜索。更重要的是，在后面一节中，我们证明了解决 NTRU 密钥恢复问题(几乎可以肯定)等同于解决某类格中的 SVP 问题。这将 NTRU 问题与一个研究得很好的问题联系起来，尽管是针对一类特殊的格。目前，使用格约化(lattice reduction)是从公钥中恢复 NTRU 私钥的最佳方法。格约化是最好的方法吗？正如整数分解和其他密码系统所依赖的各种离散对数问题一样，没有人能确定是否存在更快的算法。因此，判断 NTRU 密钥恢复问题难度的唯一方法是目前它已经被数学界和密码学界深入地研究了。然后通过应用当前已知最快的算法，可以定量估计解决该问题的难度。

如果 Eve 尝试对所有可能的私钥进行暴力搜索，这会有多难呢？注意，Eve 可以通过验证 $f(x) \star h(x) \pmod q$ 是否为三进制多项式来确定她是否找到了私钥 $f(x)$。很有可能，唯一具有此属性的多项式是 $f(x)$ 的旋转，但如果 Eve 恰好找到另一个具有此属性的三进制多项式，它也可以用作解密密钥。

于是，我们需要计算三进制多项式集合的大小。我们可以首先选择 $d_1$ 个系数为 1，再从剩余的系数中选择 $d_2$ 个为 -1，从而确定 $\mathcal{T}(d_1,d_2)$ 元素的个数。
$$
\#\mathcal{T}(d_1,d_2)={N\choose d_1}{N-d_1\choose d_2}=\frac{N!}{d_1!d_2!(N-d_1-d_2)!}.
$$
上式去到最大值时当 $d_1=d_2=N/3$。

对于暴力搜索方法，Eve 必须尝试 $\mathcal{T}(d+1,d)$ 中每个多项式，直到找到解密密钥，但注意到 $f(x)$ 的旋转都是解密密钥，所以共有 $N$ 个可能结果。因此 Eve 大约要做 $\#\mathcal{T}(d+1,d)/N$ 次尝试。

*Example.* 我们考虑下面这种 NTRU：
$$
 (N,p,q,d) = (251,3,257,83).
$$
这个 NTRU 参数组合并不满足 $q>(6d+1)p$ 的要求，所以会有解密失败的可能。Eve 期望的搜索次数大约是：
$$
\frac{\mathcal{T}(84,83)}{251}=\frac{1}{251}{251\choose 84}{167\choose 83}\approx 2^{381.6}.
$$

> 教材这里还简单介绍了如何使用碰撞算法，但我实在没看懂。。。不过根据生日攻击的原理，大约需要 $\sqrt{2^{381.6}}\approx2^{190.8} $ 次搜索，也能得到相同的结论。

总的来说，我们令 $d\approx N/3$ 使 $\mathcal{T}(d+1,d)$ 取得最大值，并利用 Stirling 公式来估计碰撞搜索的次数：
$$
\#\mathcal{T}(d_1,d_2)\approx\frac{N!}{((N/3!))^3}\approx\left(\frac{N}{e}\right)^N\cdot\left(\left(\frac{N}{3e}\right)^{N/3}\right)^{-3}\approx 3^N.
$$
在这种情况下，碰撞搜索大约需要 $O(3^{N/2}/\sqrt{N})$ 次。

*Remark.* 前面说到，只有$f(x)$ 与其旋转有可能是 $\mathcal{T}(d+1,d)$ 的解密密钥。为了说明这一点，我们要求的是一些随机的 $f(x)\in \mathcal{T}(d+1,d)$ 具有以下性质的概率
$$
f(x)\star h(x) \pmod q\quad\text{is a ternary polynomial}.
$$
将上式的系数视为独立随机变量，且在模 $q$ 意义下均匀分布。任何特定系数为三元的概率为 $3/q$，因此每个系数都是三元的概率近似为 $(3/q)^N$。所以：
$$
\begin{align}
\left(\begin{array}\\
\text{Expected number of decryption}\\
\text{keys in }\mathcal{T}(d+1,d)
\end{array}
\right)&\approx \Pr\left(\begin{array}\\
f(x)\in\mathcal{T}(d+1,d)\\
\text{is a decryption key}
\end{array}
\right)\times \#\mathcal{T}(d+1,d)\\
&=(\frac{3}{q})^N {N\choose d+1}{N-d-1\choose d}.
\end{align}
$$
回到 $\mathcal{T}(84,83)$ 的例子里，$N=251,q=257$，那么：
$$
(\frac{3}{257})^N {251\choose 84}{167\choose 83}\approx2^{-1222.02}.
$$
当然，如果 $h(x)$ 是 NTRU 的公钥，则一定存在解密密钥，因为 $h(x)$ 就是由 $f(x)$ 构造得到的。但是上式计算得到的概率使得除了 $f(x)$ 与其旋转之外，不可能再由其他的解密密钥了。

## NTRU as a lattice cryptosystem

> 这一节对应教材的 6.11.

这一节，我们解释了如何将 NTRU 密钥恢复问题转化为在某种特殊类型的格中寻找最短向量的问题。教材的练习 6.32 描述了将 NTRU 明文恢复问题转化为最近向量问题。

### The NTRU lattice

设
$$
h(x)=h_0+h_1x+\dots+h_{N-1}x^{N-1}
$$
是 NTRU 的公钥。与 $h(x)$ 相关联的 NTRU 格(NTRU lattice) $L_h^{\text{NTRU}}$ 是由下面的矩阵的行扩张成的 $2N$ 维的格。
$$
M_h^{\text{NTRU}}=
\begin{pmatrix}
\begin{array}{cccc|cccc}
1&0&\cdots&0&h_0&h_1&\cdots&h_{N-1}\\
0&1&\cdots&0&h_{N-1}&h_0&\cdots&h_{N-2}\\
\vdots&\vdots&\ddots&\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&1&h_1&h_2&\cdots&h_0\\
\hline
0&0&\cdots&0&q&0&\cdots&0\\
0&0&\cdots&0&0&q&\cdots&0\\
\vdots&\vdots&\ddots&\vdots&\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&0&0&0&\cdots&q
\end{array}
\end{pmatrix}
$$
$M_h^{\text{NTRU}}$ 其实是包含了 4 个 $N\times N$ 的方块：

1. 左上方块 = 单位矩阵(Identity matrix)
2. 左下方块 = 全 0 矩阵(Zero matrix)
3. 右下方块 = $q$ 倍单位阵($q$ times the identity matrix)
4. 右上方块 = $h(x)$ 系数的循环置换

因此经常将 NTRU 矩阵缩写为：
$$
M_h^{\text{NTRU}}=\begin{pmatrix}
I & h\\
0 & qI
\end{pmatrix},
$$
我们将其看作 $2\times 2$ 的系数取自 $R$ 的矩阵。

我们将区分每一对 $R$ 中的多项式
$$
a(x)=a_0+a_1x+\dots+a_{N-1}x^{N-1}\quad\text{and}\quad b(x)=b_0+b_1x+\dots+b_{N-1}x^{N-1}
$$
通过用一个 $2N$ 维的向量：
$$
(a,b)=(a_0,a_1,\dots,a_{N-1},b_0,b_1,\dots,b_{N-1})\in \Z^{2N}.
$$
我们现在假设 NTRU 公钥 $h(x)$ 是用私有的多项式 $f(x),g(x)$ 创建的，并观察当我们用一个精心挑选的向量乘以 NTRU 矩阵时会发生什么。

**Proposition.** 假设 $f(x)\star h(x)\equiv g(x)\pmod{q}$，令 $u(x)\in R$ 满足：
$$
f(x)\star h(x)=g(x)+qu(x).
$$
 那么有
$$
(f,-u)M_h^{\text{NTRU}}=(f,g),
$$
因此向量 $(f,g)$ 就在 NTRU 格 $L_h^{\text{NTRU}}$ 中。

*Proof.* 利用 NTRU 矩阵的 $2\times 2$ 表示：
$$
(f,-u)\begin{pmatrix}
1 & h\\
0 & q
\end{pmatrix}=(f,f\star h -qu)=(f,g)
$$
**Proposition.** 设 $(N,p,q,d)$ 是 NTRU 参数，简单起见我们假设
$$
d\approx N/3\quad\text{and}\quad q\approx6d\approx2N.
$$
令 $L_h^{\text{NTRU}}$ 是一个 NTRU 格，其私钥为 $(f,g)$。

1. $\det(L_h^{\text{NTRU}})=q^N$.

2. $\left\lVert(f,g) \right\rVert\approx\sqrt{4d}\approx\sqrt{4N/3}\approx 1.155\sqrt{N}$.

3. Gauss 启发式方法(Gaussian heuristic)预测 NTRU 格中的最短向量长度为：
   $$
   \sigma(L_h^{\text{NTRU}})\approx \sqrt{Nq/\pi e}\approx0.484N.
   $$
   因此，当 $N$ 很大时，有很大概率 $L_h^{\text{NTRU}}$ 格中的最短向量就是 $(f,g)$ 以及其旋转(rotations)。更进一步我们有：
   $$
   \frac{\left\lVert(f,g) \right\rVert}{\sigma(L)}\approx\frac{2.39}{\sqrt{N}}
   $$
   所以向量 $(f,g)$ 比 Gauss 启发式方法预测的更短，相差一个 $O(1/\sqrt N)$ 的因子。

*Proof.* 

1. 在格的基本定义那一小节我们证明过，格的行列式就等于对应的基本域的体积，亦即基底构成的矩阵的行列式。于是在这里，$\det(L_h^{\text{NTRU}})$ 就等于矩阵 $M_h^{\text{NTRU}}$ 的行列式。这是一个上三角矩阵，因此其行列式就等于对角线元素的乘积，即 $q^N$.

2. $f\in \mathcal{T}(d+1,d),g\in \mathcal{T}(d,d)$，因此 $f$ 和 $g$ 都是大约有 $d$ 个系数为 1，$d$ 个系数为 -1.

3. 利用命题 1 和格的维度为 $2N$ 的条件，我们可以估计 Gauss 期望最短长度为：
   $$
   \sigma(L_h^{\text{NTRU}})=\sqrt{\frac{2N}{2\pi e}}(\det(L))^{\frac{1}{2N}}=\sqrt{\frac{Nq}{\pi e}}\approx \sqrt{\frac{2}{\pi e}}N.
   $$

### Quantifying the security of an NTRU lattice

上面的命题是说 Eve 如果能够找到 NTRU 格 $L_h^{\text{NTRU}}$ 中的最短向量，那么她就能够确定 Alice 的 NTRU 私钥。因此 NTRU 的安全性至少依赖于解决格 $L_h^{\text{NTRU}}$ 上的 SVP 的困难程度。更一般地说，如果 Eve 能在约为 $N^{\epsilon}$ 的因子范围内解决 $L_h^{\text{NTRU}}$ 上的 apprSVP，$\epsilon < \frac{1}{2}$，那么她寻找到的短向量就很有可能是解密密钥。

这就引出了一个问题：关于估计寻找 NTRU 格中的最短向量的困难程度。LLL 算法可以在多项式时间内以 $2^N$ 因子范围解决 apprSVP。但是如果 $N$ 很大，那么 LLL 就不能找到 $L_h^{\text{NTRU}}$ 中非常小的向量。后面我们将会介绍 LLL 算法的一个推广，称为 BKZ-LLL，能够找到非常小的向量。BKZ-LLL 算法包括一个块大小参数 $\beta$，能够以 $\beta^{2N/\beta}$ 的因子范围解决 apprSVP，但是其运行时间是 $\beta$ 的指数级别。

不幸的是，标准的格约化算法(如 BKZ-LLL)的运行特性远没有被理解透彻。这使得理论上很难预测格约化算法在任何给定的格上的表现。因此在实践中，基于格的密码系统(如 NTRU)的安全性必须通过实验来确定。

大体上，人们会选取一系列参数 $(N,q,d)$，其中 $N$ 逐渐增大，而涉及 $N,q,d$ 的某些比值被保持在大致不变的水平。对于每组参数,人们使用不断增大的块大小 $\beta$ 运行多次 BKZ-LLL 实验，直到算法在 $L_h^{\text{NTRU}}$ 中找到一个短向量。然后将平均运行时间的对数值绘制成关于 $N$ 的图，验证这些点大致位于一条直线上，并计算最佳拟合直线：
$$
\log(\text{Running Time})=AN+B.
$$
在对许多 $N$ 值进行上述实验直到计算变得不可行之后，我们可以使用上式描绘的直线来推测在更大的 $N$ 值情况下，在 NTRU 格 $L_h^{\text{NTRU}}$ 中找到私钥向量所需的预期时间。这种实验表明，当 $N$ 值在 250 到 1000 的范围内时，NTRU 具有与当前安全应用的RSA，ElGamal 和 ECC 相当的安全级别。

*Remark*. NTRU 格中的目标短向量比高斯启发式预测的要短 $O(\sqrt{N})$ 倍。理论上和实验上都是如此，如果一个 $n$ 维格中存在一个极小的向量，比如比高斯预测小 $O(2^n)$ 倍，那么格约化算法如 LLL 及其变体就非常擅长找到这个微小向量。一个自然而有趣的问题是，如果向量只比高斯预测小 $O(n^\epsilon)$ 倍，那么它们是否也同样容易被找到？目前还没有人知道这个问题的答案。



























