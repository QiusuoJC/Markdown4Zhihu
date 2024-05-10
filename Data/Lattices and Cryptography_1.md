#  Lattices and Cryptography

[TOC]

本专栏内容翻译自《An Introduction to Mathematical Cryptography》的 Chapter 6: Lattices and Cryptography，[原书链接](https://link.springer.com/book/10.1007/978-1-4939-1711-2)。在翻译学习过程中，也穿插了一些我个人的理解。学习格有很多好的资料，我也是无意中找到的这本教材，选择它是因为书中本章节开头的引言部分吸引了我继续学下去，并且本书的例子也非常多。于是就把整个第六章节给学习了一遍。本教材关于格的内容主要是介绍了最短向量问题(SVP)、最近向量问题(CVP)，GGH 和 NTRU 密码系统，以及 LLL 格约化算法。如果你想学习 LWE 有关的知识，那么这本书可能让你失望了。此外，如果你对线性代数的知识有所遗忘，书中也贴心的带领你简单回顾了一下。最后，本人能力十分之有限，做翻译学习的目的也是想入门格密码，其中难免会出现很多翻译以及理解上的问题，感谢大家的批评指正。下面进入正题。

回忆一下，一个定义在实数域 $\mathbb{R}$ 上的向量空间 $V$​ 是一个向量的集合，满足加法和数乘封闭。一个格(lattice)和向量空间非常相似，只是我们限制与向量做数乘运算的必须是整数(因此产生的是一些离散的点)。这看起来细微的限制却产生了许多有趣且微妙的问题。我们首先复习一下线性代数和向量空间的知识，再引出格的知识。

## A congruential public key cryptosystem

> 这一节对应教材的 6.1.

这一节我们介绍一个真实的公钥密码系统的简单模型(toy model)。这个版本与二维格有关，因此存在致命的弱点，因为其维度太低。然而，它作为一个示例，说明了即使底层的困难问题看似与格无关，格在密码分析中也可能出现。此外，它提供了对 NTRU 公钥密码系统的最低维度介绍，该系统将在后面描述。

Alice 首先选择一个大的正整数 $q$ 作为公共参数，然后选择另外两个秘密的正整数 $f$ 和 $g$，满足：
$$
f<\sqrt{q/2},\quad \sqrt{q/4}<g<\sqrt{q/2},\quad\text{and}\quad\gcd(f,q)=1.
$$
Alice 随后计算：
$$
h\equiv f^{-1}g\pmod{q}\quad\text{with}\quad0<h<q.
$$
这里 $f^{-1}$ 是 $f$ 关于模 $q$ 的逆元。注意到与 $q$ 相比 $f,g$ 都比较小，都是 $O(\sqrt{q})$ 的数量级，而 $h$ 是 $O(q)$的数量级，相对更大一些。Alice 的私钥是小整数 $f,g$，公钥是大整数 $h$。

为了发送消息，Bob 选择明文 $m$ 和随机数 $r$ (临时密钥，ephemeral key)满足不等式：
$$
0<m<\sqrt{q/4}\quad\text{and}\quad 0<r<\sqrt{q/2}.
$$
他计算密文：
$$
e\equiv rh+m\pmod{q}\quad\text{with}\ 0<e<q
$$
并发送给 Alice.

Alice 进行解密，首先计算：
$$
a\equiv fe\pmod{q}\quad\text{with}\ 0<a<q,
$$
再计算
$$
b\equiv f^{-1}a\pmod{g}\quad\text{with}\ 0<b<g.
$$
注意这一步的 $f^{-1}$ 是 $f$ 关于模 $g$ 的逆元。

我们现在验证 $b=m$，即通过这种方式 Alice 能够成功恢复出 Bob 的明文。观察到 $a$ 满足：
$$
a\equiv fe\equiv f(rh+m)\equiv frf^{-1}g+fm\equiv rg+fm\pmod{q}.
$$
对 $f,g,r,m$ 大小的限制表明了 $rg+fm$ 是小的整数：
$$
rg+fm<\sqrt{\frac{q}{2}}\sqrt{\frac{q}{2}}+\sqrt{\frac{q}{2}}\sqrt{\frac{q}{4}}<q.
$$
因此 $0<a<q$，同余式 $a\equiv fe\pmod{q}$ 便转化为了等式
$$
a=fe=rg+fm.
$$
最后 Alice 计算：
$$
b\equiv f^{-1}a\equiv f^{-1}(rg+fm)\equiv f^{-1}fm\equiv m \pmod{g}\quad\text{with}\ 0<b<g.
$$
因为 $m<\sqrt{q/4}<g$，则 $b=m$。下图总结了同余密码系统(congruential cryptosystem)。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\21.png" style="zoom:67%;"/>
      <figcaption>图 1. 同余公钥密码系统，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

*Example.* Alice 选择
$$
 q =122430513841,\quad f= 231231,\quad \text{and}\quad g = 195698.
$$
这里 $f\approx 0.66\sqrt{q},g\approx 0.56\sqrt{q}$ 都是被允许的值。Alice 计算：
$$
f^{−1} \equiv 49194372303 \pmod{q} \quad\text{and} \quad h\equiv f^{−1}g \equiv 39245579300 \pmod{q}.
$$
Alice 的公钥为 $(q,h)=(122430513841,39245579300)$。

Bob 决定发送给 Alice 明文 $m=123456$，随机数为 $r=101010$。他使用 Alice 的公钥来计算密文：
$$
e\equiv rh+m\equiv 18357558717\pmod{q},
$$
并将密文发送给 Alice。

为了解密 $e$，Alice 首先用 $f$ 计算：
$$
a\equiv fe\equiv48314309316 \pmod{q}.
$$
注意到 $a=48314309316 < 122430513841 = q$。她再利用 $f^{-1}\equiv 193495 \pmod{g} $ 来计算：
$$
f^{−1}a \equiv 193495\cdot48314309316 \equiv 123456 \pmod{g},
$$
与理论相符，结果就是 Bob 的明文。

Eve 如何攻击这个系统呢？她可能尝试暴力搜索所有可能的私钥值或者所有可能的明文值，但是这都需要 $O(q)$ 数量级的操作。让我们更详细地考虑一下 Eve 的任务如果她尝试从已知的公钥 $(q,h)$ 中得到私钥 $(f,g)$。不难看出如果 Eve 能够找到任意的正整数对 $F,G$ 满足：
$$
Fh\equiv G\pmod{q}\quad\text{and}\quad F=O(\sqrt{q})\quad\text{and}\quad G=O(\sqrt{q}),
$$
那么 $(F,G)$ 就很有可能是解密密钥。重写上面的同余式为等式 $Fh=G+qR$，我们将 Eve 的任务重新表述为寻找一对相对较小的整数 $(F, G)$​，满足以下性质：
$$
F\underbrace{(1,h)}_{\text{known}}-R\underbrace{(0,q)}_{\text{known}}=\overbrace{(F,G)}^{\text{unknown}}.
$$
$F,R$ 是未知的整数，$(1,h),(0,q)$ 是已知的向量，$(F,G)$ 是未知的小向量。

因此 Eve 知道两个向量 $v_1=(1,h),v_2=(0,q)$，每一个的长度都是 $O(q)$，她想要找到一个线性组合 $w=a_1 v_1+a_2v_2$ 满足 $w$ 的长度为 $O(\sqrt{q})$，但是注意 $a_1,a_2$ 都需要时整数，因此 Eve 需要在下面的向量集合中找到一个短的非零向量：
$$
L=\{a_1v_1+a_2v_2: a_1,a_2\in\Z\}.
$$
这个集合 $L$ 就是 2 维格的一个例子。它看起来非常像由基底 $\{v_1,v_2\}$ 生成的 2 维向量空间，除了线性组合的系数只能取整数。

在上面的例子中，寻找 2 维格中的短向量存在非常快速的方法求解。这个方法是由 Gauss 提出的，在格约化一节中有描述。

## Subset-sum problems and knapsack cryptosystems

> 这一节对应教材的 6.2.

第一个基于 $\mathcal{NP}$ 完全问题的密码系统是由 Merkle 和 Hellman 在20世纪70年代末设计的。他们使用了以下数学问题的一个变体，该问题是经典背包问题的推广。

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\22.png" style="zoom:67%;"/>
      <figcaption>图 2. 子集求和问题，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

*Example.* 令 $M=(2,3,4,9,14,23),S=27$。那么经过几次尝试就能得到子集合 $\{4,9,14\}$ 的和是 $27$，不能验证这是唯一的子集合满足和为 $27$。类似地，如果我们取 $S=29$，那么我们能够找到 $\{2,4,23\}$ 满足条件，但是这种情况还有第二个解 $\{2,4,9,14\}$。

还有另一种描述子集求和问题的方式。序列：
$$
 M=(M_1,M_2,\dots,M_n)
$$
是公开的正整数。Bob 选择一个秘密的二进制向量(binary vector) $x=(x_1,x_2,\dots,x_n)$，即每一个 $x_i$ 要么取 0 要么取 1。Bob 计算和：
$$
S=\sum_{i=1}^n x_i M_i
$$
并将 $S$ 发给 Alice。子集和问题要求 Alice 要么找到原始的向量 $x$，要么找到另一个二进制向量，能够给出相同的和。注意向量 $x$ 告诉 Alice 哪一个 $M_i$ 被包含进 $S$，因为 $M_i$ 在和式 $S$ 中当且仅当 $x_i=1$。因此只要确定了二元向量 $x$ 就是确定了 $M$ 的一个子集合。

显然，Alice 能够遍历所有 $2^n$ 个 $n$ 长的二元向量来找到 $x$。一个简单的碰撞算法可以将复杂度的指数减半。

**Proposition.** 令 $M=(M_1,M_2,\dots,M_n)$ 并令 $(M,S)$ 表示一个子集和问题。对于所有满足下面条件的整数集合 $I$ 和 $J$：
$$
I\subset \{i:1 \leq i\leq \frac{1}{2}n\}\quad\text{with}\quad J\subset \{j:\frac{1}{2}n < j \leq n\}
$$
计算得到两个序列：
$$
A_I=\sum_{i \in I}M_i\quad\text{and}\quad B_J=S-\sum_{j\in J}M_j.
$$
那么这些序列包含一对集合 $I_0,J_0$ 满足 $A_{I_0}=B_{J_0}$，并且集合 $I_0,J_0$ 给出了一个子集和问题的解：
$$
S=\sum_{i \in I_0}M_i+\sum_{j \in J_0}M_j.
$$
每个序列的条目最多有 $2^{n/2}$ 项，所以算法的运行时间为 $O(2^{n/2+\epsilon})$，其中 $\epsilon$ 是序列的排序和比较所引入的小值。

*Proof.* 如果 $x$ 是一个二进制向量，它是给定的子集和问题的一个解，那么我们可以将该解写作:
$$
\sum_{1\leq i\leq \frac{1}{2}n}x_iM_i=S-\sum_{\frac{1}{2}n< i\leq n}x_iM_i.
$$
子集合 $I,J$ 的个数均为 $O(2^{n/2})$。

如果 $n$​ 很大，那么一般来说解决一个随机的子集和问题的实例是很困难的。然而，我们假设 Alice 拥有关于 $M$ 的某些秘密知识或陷门(trapdoor)信息，使得她能够确保解 $x$ 是唯一的，并且允许她很容易地找到 $x$。那么 Alice 可以将子集和问题用作公钥加密系统。Bob 的明文是向量 $x$，他的加密消息是和 $S = \sum x_iM_i$，只有Alice可以轻松从已知的 $S$ 中恢复 $x$。

但是，Alice 可以使用什么技巧来确保她可以解决这个特定的子集和问题而其他人却不能呢？一种可能性是使用一个极其容易解决的子集和问题，但以某种方式对其他人隐藏了这个简单的解决方案。

**Definition.** 一个整数的超递增序列(superincreasing sequence)是正整数序列 $r=(r_1,r_2,\dots,r_n)$ 满足性质：
$$
r_{i+1}\geq 2r_i\quad\text{for all }1\leq i\leq n-1.
$$
下面的估计解释了这个序列的名字。

**Lemma.** 设 $r=(r_1,\dots,r_n)$ 是一个超递增序列。那么
$$
r_k > r_{k-1}+\dots+r_2+r_1 \quad\text{for all }2\leq k\leq n. 
$$
*Proof.* 我们通过对 $k$ 进行归纳来给出证明。对于 $k=2$ 我们有 $r_2\geq 2r_1 >r_1$，是归纳基础。现在我们假设上面的引理对于 $2\leq k< n$ 是正确的，那么用超递增序列的性质和归纳假设，我们能够得到：
$$
r_{k+1}\geq 2r_k=r_k+r_k >r_k +(r_{k−1} +···+r_2 +r_1).
$$
这就证明了引理对于 $k+1$ 也是正确的。

当 $M$ 中的整数形成一个超递增序列时，子集和问题是非常容易解决的。

**Proposition.** 设 $(M,S)$ 是一个子集和问题，其中 $M$ 中的整数形成一个超递增序列。假设存在一个解 $x$，那么它是唯一的且能够用下面的算法快速计算得到：

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\23.png" style="zoom:67%;"/>
      <figcaption>图 3. 快速计算SSP的一个解，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>
*Proof.* $M$ 是一个超递增序列意味着 $M_{i+1}\geq 2M_i$。我们知道存在一个解，因此为了将它与算法产生的向量 $x$ 区分开，我们将实际的解称为 $y$。因此我们假设 $yM=S$，我们需要证明 $x=y$。

我们用逆向归纳法(downward induction)证明 $x_k=y_k$ 对于所有的 $1\leq k\leq n$ 都成立。我们的归纳假设是对于 $k<i\leq n$，$x_i=y_i$，我们需要证明 $x_k=y_k$。注意我们允许 $k = n$，在这种情况下，我们的归纳假设是自动成立的。根据归纳假设，当我们从 $i=n$ 开始执行算法到 $i=k+1$ 时，在每一阶段我们都有 $x_i=y_i$。因此在执行第 $i=k$ 次循环时，$S$ 的值已经被化简为：
$$
S_k=S-\sum_{i=k+1}^n x_iM_i=\sum_{i=1}^ny_iM_i-\sum_{i=k+1}^n x_iM_i=\sum_{i=1}^k y_iM_i.
$$
当执行第 $i=k$ 次循环时，可能有两种情况发生：
$$
\begin{align}
(1)\quad &y_k=1 \ \Longrightarrow\ S_k\geq M_k &\Longrightarrow\ &x_k=1,\quad \sqrt{}\\
(2)\quad &y_k=1 \ \Longrightarrow\ S_k\leq M_{k-1}+\dots+M_1<M_k &\Longrightarrow\ &x_k=0,\quad \sqrt{}
\end{align}
$$
情况(2)我们使用了上面的引理推导出 $M_{k-1}+\dots+M_1$ 严格小于 $M_k$，这两种情况我们都能得到 $x_k=y_k$，这就完成了 $x=y$ 的证明。此外，这也表明解是唯一的，因为我们已经证明了任何解都与算法的输出一致，而算法本质上对于任何给定的输入 $S$ 都返回一个唯一的向量 $x$。

*Example.* 集合 $M=(3,11,24,50,115)$ 是超递增的。我们令 $S=142$ 是 $M$ 中一些元素的和。首先 $S\geq 115$，所以 $x_5=1$，我们将 $S$ 替换为 $S-115=27$。接下来 $27<50$，所以 $x_4=0$。继续，$27\geq 24$，所以 $x_3=1$，此时 $S$ 变为了 $27-24=3$。$3<11$，所以 $x_2=0$。最后 $3\geq 3$，所以 $x_1=1$。注意 $S$ 已经被减为 $3-3=0$ 了，于是 $x=(1,0,1,0,1)$ 就是一个解。检查我们的答案：
$$
1\cdot3+0\cdot11+1\cdot24+0\cdot50+1\cdot115 = 142.\quad \sqrt{}
$$
Merkle 和 Hellman 提出了一个基于超递增子集和问题的公钥密码系统，该系统使用同余来伪装(disguise)。为了创建公/私钥对，Alice 选择一个超递增序列 $r = (r_1,\dots,r_n)$。再选择两个大的秘密整数 $A$ 和 $B$ 满足：
$$
B>2r_n\quad\text{and}\quad\gcd(A,B)=1.
$$
Alice 再构造一个新的非超递增序列 $M$，其中：
$$
M_i \equiv Ar_i \pmod{B}\quad\text{with }0\leq M_i <B.
$$
序列 $M$ 就是 Alice 的公钥。

为了加密消息，Bob 选择二进制向量 $x$ 作为明文，计算密文发送给 Alice:
$$
S=x\cdot  M=\sum_{i=1}^nx_iM_i.
$$
为了解密，Alice 首先计算：
$$
S'\equiv A^{-1}S\pmod{B}\quad\text{with }0\leq S'<B.
$$

然后，Alice 用超递增序列 $r$ 和上面命题提到的快速算法来解决 $S'$ 的子集和问题。

这样做能够成功解密是因为：
$$
S'\equiv A^{-1}S\equiv A^{-1}\sum_{i=1}^nx_iM_i\equiv A^{-1}\sum_{i=1}^nx_iAr_i\equiv\sum_{i=1}^n x_ir_i\pmod{B}.
$$
根据假设 $B>2r_n$ 和引理：
$$
\sum_{i=1}^nx_ir_i \leq \sum_{i=1}^nr_i<2r_n<B,
$$
所以同余式就变成了等式 $S'=\sum x_ir_i$，$S'$ 的取值范围是 $0$ 到 $B-1$。

下图总结了 Merkle-Hellman 密码系统：

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\24.png" style="zoom:67%;"/>
      <figcaption>图 4. Merkle-Hellman 子集求和密码系统，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

*Example.* 设 $ r =(3,11,24,50,115)$ 是 Alice 的秘密超递增序列，$A=113,B=250$ 是她所选的秘密大整数。那么她的伪装序列为：
$$
\begin{align}
M &\equiv (113\cdot3,113\cdot11,113\cdot24,113\cdot50,113\cdot115) \pmod{250}\\
&=(89,243,212,150,245).
 \end{align}
$$
注意到 $M$ 序列远不是超递增的(即使她重新排列项，使其递增)。
Bob 决定向 Alice 发送秘密信息 $x = (1,0,1,0,1)$。他加密 $x$ 通过计算：
$$
S=x\cdot M=1\cdot89+0\cdot243+1\cdot212+0\cdot150+1\cdot245=546.
$$
在收到 $S$ 后，Alice 将其乘以 177，即 113 模 250 的逆，得到：
$$
S'\equiv 177\cdot 546 = 142 \pmod{250}.
$$
随后，Alice 用命题的快速算法来求解 $S'=x\cdot r$，恢复出明文 $x$。

基于伪装子集和问题的加密系统被称为子集和密码系统(subset-sum cryptosystems)或背包密码系统(knapsack cryptosystems)。其基本思想是从一个秘密的超递增序列开始，使用秘密的模线性运算对其进行伪装，并将伪装后的序列作为公钥发布。Merkle 和 Hellman 最初的系统建议对 $Ar\pmod{B}$ 的条目应用秘密置换作为额外的安全层。后来的版本由许多人提出，涉及对一些不同的模数进行多次乘法和约减。关于背包加密系统的优秀综述，可以参阅 Odlyzko 的文章(The Rise and Fall of Knapsack Cryptosystems)。

*Remark.* 关于背包系统必须考虑的一个重要问题是达到期望的安全级别所需的各种参数的大小。有 $2^n$ 个二进制向量 $x=(x_1,\dots,x_n)$，我们已经在本节的第一个命题中看到，存在一个碰撞算法，因此可以在 $O(2^{n/2})$ 次运算中破解背包加密系统。因此，为了获得 $2^k$ 数量级的安全性，有必要取 $n>2k$，例如，$2^{80}$ 安全性需要 $n>160$。尽管这提供了针对碰撞攻击的安全性，但并不排除存在其他更有效攻击的可能性，我们将在最后一节看到，这些攻击实际上是存在的。

*Remark.* 假设我们已经选择了 $n$，那么其他参数应该选择多大的呢？事实证明，如果 $r_1$ 太小，就会有简单的攻击方法，所以必须要令 $r_1>2^n$。序列的超递增性质意味着：
$$
r_n > 2r_{n−1} > 4r_{n−1} > \dots> 2^nr_1 > 2^{2n}.
$$
那么 $B>2r_n=2^{2n+1}$，所以我们得到了公钥 $M_i$ 和密文 $S$ 满足：
$$
M_i=O(2^{2n})\quad\text{and}\quad S=O(2^{2n}).
$$
因此,公钥 $M$ 是一个包含 $n$ 个整数的序列，每个整数大约有 $2n$ 位长，而明文 $x$ 包含 $n$ 位信息，密文大约有 $2n$ 位。注意到消息的扩展比为 $2:1$。
例如，假设 $n=160$。那么公钥大小大约为 $2n\cdot n=51200$ 位。与 RSA 或 Diffie-Hellman 相比，对于 $2^{80}$ 数量级的安全性，它们的公钥大小大约只有 1000 位。这个较大的密钥尺寸可能看起来是一个很大的缺点，但它会被背包系统极快的处理速度所弥补。事实上，背包系统的解密只需要一次(或很少几次)的模乘运算，而加密根本不需要。这比 RSA 和 Diffie-Hellman 使用大量计算密集型的模幂运算要高效得多。从历史上看，这使得背包密码系统具有非常大的吸引力。

*Remark.* 已知的解决随机选择的子集和问题的最佳算法是碰撞算法，如前面命题已经提到的。不幸的是，**随机选择的子集和问题没有陷门，因此不能用于创建密码系统。**事实证明，使用伪装的超递增子集和问题有其他更有效的攻击算法。第一种此类攻击由 Shamir、Odlyzko、Lagarias 等人使用各种特设的方法(ad hoc methods)，但在 1985 年著名的 LLL 格约化论文发表之后，很明显，基于背包的加密系统存在根本缺陷。粗略地说，如果 $n$ 小于 300 左右，那么格约化允许攻击者在短时间内从密文 $S$ 中恢复明文 $x$。因此，一个安全的系统需要 $n>300$，在这种情况下，私钥长度大于 $2n^2=180000\text{ bits}\approx 176\text{ KB}$。这就太大了，以至于使安全的背包系统变得不切实际。

我们现在简要描述 Eve 如何使用向量来重新表述子集和问题。假设她想把 $S$ 写成集合 $M =(m_1,...,m_n)$ 的子集和。她的第一步是形成矩阵：
$$
\begin{pmatrix}
2& 0& 0 &\cdots& 0& m_1\\
0& 2& 0 &\cdots& 0& m_2\\
0& 0& 2 &\cdots& 0& m_3\\
\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0& 0& 0 &\cdots& 2& m_n\\
1& 1& 1 &\cdots& 1& S\\
\end{pmatrix}.\tag{*}
$$
> 我把这个矩阵是称为 $(*)$ 矩阵是为了方便后面文章的叙述。

相关的向量是上面矩阵的行，她标记为：
$$
\begin{align}
v_1&=(2,0,0,\dots,0,m_1),\\
v_2&=(0,2,0,\dots,0,m_2),\\
\vdots&\qquad\qquad\quad\vdots\\
v_n&=(0,0,0,\dots,2,m_n),\\
v_{n+1}&=(1,1,1,\dots,1,S).\\
\end{align}
$$
与同余密码系统中考虑的 2 维示例类似，Eve 考虑 $v_1,\dots,v_{n+1}$ 所有整系数线性组合的集合，
$$
L=\{a_1v_1+a_2v_2+\dots+a_nv_n+a_{n+1}v_{n+1}:a_1,a_2,\dots,a_{n+1}\in \Z\}
$$
集合 $L$ 就是格的另一个例子。

假设 $x=(x_1,\dots,x_n)$ 是一个子集和问题的解，那么格 $L$ 包含向量：
$$
\begin{align}
t&=\sum_{i=1}^nx_iv_i-v_{n+1}\\
&=(2x_1,0,\dots,0,m_1x_1)+\dots+(0,0,\dots,2x_n,m_nx_n)-(1,1,\dots,1,S)\\
&=(2x_1-1,2x_2-1,\dots,2x_n-1,0),
\end{align}
$$
$t$ 的最后一个坐标为 $0$ 是因为 $S=x_1m_1+\dots+x_nm_n$。

我们现在进入了问题的关键部分。由于 $x_i$ 都是 0 或 1，所有的 $2x_i-1$ 值都是 $\pm1$，因此向量 $t$ 非常短，$\left\lVert t\right\rVert = \sqrt{n}$。另一方面，我们已经看到 $m_i = O(2^{2n}), S=O(2^{2n})$，所以生成 $L$ 的向量都有长度 $v_i = O(2^{2n})$。因此，除了 $t$ 之外，$L$ 不太可能包含任何长度小到 $\sqrt{n}$ 的非零向量。如果我们假设 Eve 知道一种算法，可以在格中找到短的非零向量，那么她将能够找到 $t$，从而恢复明文 $x$。

在格中寻找短向量的算法称为格约化算法(lattice reduction algorithms)。其中最著名的是我们之前提到的 LLL 算法及其变体,如 LLL-BKZ。

## A brief review of vector spaces

> 这一节对应教材的 6.3.

向量空间有广义的定义，不过本章我们考虑在 $\mathbb{R}^m$ 上的向量空间。

**向量空间(Vector Spaces).** 一个向量空间 $V$ 是 $\mathbb{R}^m$ 的一个子集合，满足性质：
$$
\alpha_1 v_1+\alpha_2 v_2 \in V,\quad\text{for all}\ v_1,v_2 \in V \ \text{and all}\ \alpha_1,\alpha_2 \in R
$$
即，一个向量空间 $V$ 是 $\mathbb{R}^m$ 的一个子集合，满足加法和取自 $R$ 上元素的标量乘法封闭。

**线性组合(Linear Combinations).** 令 $v_1,v_2,\dots,v_k \in V$.   $v_1,v_2,\dots,v_k \in V$ 的一个线性组合为一个有着如下形式的向量：
$$
w=\alpha_1 v_1+\alpha_2 v_2+\dots +\alpha_k v_k \quad with\ \alpha_1, \dots, \alpha_k \in \mathbb{R}
$$
所有这样的线性组合向量构成的集合：
$$
\{\alpha_1 v_1+\alpha_2 v_2+\dots +\alpha_k v_k :\ \alpha_1, \dots, \alpha_k \in \mathbb{R}\}
$$
称为 $\{v_1,v_2,\dots ,v_k \}$ 的一个扩张(span).

**线性无关(Independence).** 称一组向量 $v_1,v_2,\dots,v_k \in V$ 是线性无关(linearly independent)当且仅当：
$$
\alpha_1 v_1+\alpha_2 v_2+\dots +\alpha_k v_k=0
$$
时，$\alpha_1=\alpha_2=\dots=\alpha_k=0$. 而线性相关则是上式成立时，至少有一个 $\alpha_i$ 非0.

**基底(Bases).** $V$ 的一个基底是一组线性无关向量 $v_1,v_2,\dots,v_n$ 能够扩张为整个 $V$. 即对于每一个向量 $w \in V$ , 都存在唯一的实数 $\alpha_1, \dots, \alpha_n \in \mathbb{R}$, 使得 $w$ 可以表示为：
$$
w=\alpha_1 v_1+\alpha_2 v_2+\dots +\alpha_n v_n
$$
**Proposition.** 令 $V \subset \mathbb{R}^m$ 是一个向量空间

1. 存在一个 $V$ 的基底

2. $V$ 的任何两个基底都有相同数量的元素。基底中元素（向量）的个数称为向量空间的维数(dimension)

3. 令 $v_1,v_2,\dots,v_n$ 是 $V$ 的一个基底，$w_1,w_2,\dots,w_n$ 是 $V$ 中 $n$ 个向量。则 $w_j$ 可以写为 $v_i$​​ 的线性组合：
   $$
   w_1 = \alpha_{11}v_1 + \alpha_{12}v_2+\dots+\alpha_{1n}v_n,\\
   w_2 = \alpha_{21}v_1 + \alpha_{22}v_2+\dots+\alpha_{2n}v_n,\\
   \vdots \\
   w_2 = \alpha_{n1}v_1 + \alpha_{n2}v_2+\dots+\alpha_{nn}v_n,\\
   $$
   

   则 $w_1,w_2,\dots,w_n$ 也是 $V$ 的一组基当且仅当以下矩阵的行列式不等于0：
   $$
   \begin{pmatrix}
   {\alpha_{11}}&{\alpha_{12}}&{\cdots}&{\alpha_{1n}}\\
   {\alpha_{21}}&{\alpha_{22}}&{\cdots}&{\alpha_{2n}}\\
   {\vdots}&{\vdots}&{\ddots}&{\vdots}\\
   {\alpha_{n1}}&{\alpha_{n2}}&{\cdots}&{\alpha_{nn}}\\
   \end{pmatrix}
   $$

下面我们将介绍如何衡量向量的长度(lengths) 和角度(angles). 这与点积(dot product)和 Euclidean norm(即 $L_2$ 范式) 有关.

**Definition.** 令 $v,w \in V \subset \mathbb{R}^m$, 并用坐标分别表示：
$$
v=(x_1,\dots,x_m)\ \text{and}\ w = (y_1, \dots, y_m)
$$
$v$ 与 $w$ 的内积为：
$$
v\cdot w = x_1y_1 + \dots+x_my_m
$$
我们说 $v$ 与 $w$ 是正交的(orthogonal) 如果 $v\cdot w = 0$.

至于长度，或者说是 Euclidean norm 为：
$$
\left\lVert v \right\rVert = \sqrt{x^{2}_1+x^{2}_2+\dots +x^{2}_m}
$$
注意到点积和范数之间有以下关系：
$$
v\cdot v =\left\lVert v \right\rVert^2
$$
**Proposition.** 令 $v,w \in V \subset \mathbb{R}^m$.

1. 令 $\theta$ 表示 $v$ 与 $w$ 之间的角度。我们将向量 $v$ 和 $w$ 的起点放在原点 0，则
   $$
   v\cdot w =\left\lVert v \right\rVert \left\lVert w \right\rVert \cos(\theta)
   $$

2. (Cauchy-Schwarz inequality)
   $$
   \lvert v\cdot w \rvert \leq \left\lVert v \right\rVert \left\lVert w \right\rVert 
   $$

*Proof.* 当命题 1. 成立时，Cauchy-Schwarz 不等式立刻就能得到。但我们这里给出一个更直接的证明。

- 当 $w=0$ 时，nothing to prove. 以下考虑 $w\neq 0$.

- 考虑函数：
  $$
  \begin{align}
  f(t) = \left\lVert v -tw\right\rVert^2 &=(v-tw)\cdot(v-tw)\\
  &=v\cdot v -2tv\cdot w +t^2 w\cdot w \\
  &=\left\lVert w\right\rVert^2 \cdot t^2 -2v\cdot w \cdot t +\left\lVert v \right\rVert^2
  \end{align}
  $$
  对于 $\forall t \in \mathbb{R}$, 都有$f(t) \geq 0$, 因此我们取其最小值, 即当 $t=\frac{v\cdot w}{\left\lVert w \right\rVert^2}$ 时：
  $$
  f(\frac{v\cdot w}{\left\lVert w \right\rVert^2})=\left\lVert v\right\rVert^2 - \frac{(v\cdot w)^2}{\left\lVert w \right\rVert^2} \geq 0
  $$
  化简后开根号，即可证得。

**Definition.** 一个向量空间 $V$ 的正交基(orthogonal basis) 是一组基满足：
$$
v_i \cdot v_j = 0 \quad \forall i \neq j
$$
如果额外满足$\forall i,\ \left\lVert v_i \right\rVert=1$, 则称为标准正交基(orthonormal).

当 $v_1, \dots, v_n$ 是正交基时，基的线性组合 $v = a_1v_1+\dots+a_nv_n$ 有如下性质：
$$
\begin{align}
\left\lVert v \right\rVert^2 &= \left\lVert a_1v_1+\dots+a_nv_n \right\rVert^2\\
&=(a_1v_1+\dots+a_nv_n)\cdot (a_1v_1+\dots+a_nv_n)\\
&=\sum_{i=1}^n\sum_{j = 1}^n a_ia_j(v_i\cdot v_j)\\
&=\sum_{i=1}^n a_i^2\left\lVert v_i \right\rVert^2
\quad \text{since}\ v_i \cdot v_j = 0\ \text{for}\ i\neq j 
\end{align}
$$
若基底是标准正交基，则上式可化简为 $\left\lVert v \right\rVert^2 = \sum a_i^2$.

Gram-Schmidt 算法可以创造一个标准正交基，这里讨论通用算法的一个变体，相应给出的是正交基。

**Theorem (Gram-Schmidt Algorithm).** 令 $v_1, \dots, v_n$ 是向量空间 $V\subset \mathbb{R}^m$ 的一组基。下面算法可以构建 $V$ 的一个标准正交基 $v_1^*, \dots, v_n^*$​：

<p align="center">
  <figure>
  <img src="D:\程建勋学习\大三下\创新资助计划\LBC\img\1.png" style="zoom:67%;"/>
      <figcaption>图 5. Gram-Schmidt算法，源自《An Introduction to Mathematical Cryptography》</figcaption></figure>
      </p>

两个基底满足以下性质：
$$
\text{Span}\{v_1,\dots,v_i\} = \text{Span}\{v_1^*,\dots,v_i^*\}\quad \text{for all}\ i=1,\dots,n.
$$
> 注意对于格 $L$ 来说，Gram-Schmidt 算法生成的 $n$ 个正交基并不一定仍在 $L$ 中，即并不一定仍是格 $L$ 的基。这组基是由 $v_1, \dots, v_n$ 扩张成的向量空间的一组正交基。

*Proof.* 我们需要证明两件事：

>- 新生成的基底相互正交
>
>- 两组基底的扩张是相同的

1. 基的正交性利用**数学归纳法**来证明。假设 $v_1^*,\dots, v_{i-1}^*$ 是相互正交的，我们需要证明 $v_i^*$ 与前面所有带有"*"的向量是正交的。对于 $k<i$, 我们计算：
   $$
   \begin{align}
   v_i^* \cdot v_k^* &= \left(v_i - \sum_{j=1}^{i-1}\mu_{ij}v_j^* \right) \cdot v_k^*\\
   &=v_i \cdot v_k^* - \mu_{ij} \left\lVert v_k^* \right\rVert^2 &\quad \text{since}\ v_k^* \cdot v_j^* = 0 \ \text{for}\ j \neq k,\\
   &=0 &\quad \text{将}\ \mu_{ij}\ \text{根据定义}代入即可得到
   \end{align}
   $$

2. 扩张本质是集合，而证明集合的相等我们只需证明**相互包含**即可。

   - $\subseteq$：根据 $v_i^*$ 的定义，我们可以将 $v_i$ 表示为：
     $$
     v_i = v_i^*+\sum_{j=1}^{i-1}\mu_{ij}v_j^*
     $$
     即 $v_i$ 可以被表示为 $v_i^*$ 的线性组合。于是有 $v_i \in \text{Span}\{v_1^*,\dots,v_i^*\}$. 即有 $\subseteq$

   - $\supseteq$：数学归纳法。假设 $v_1^*,\dots,v_{i-1}^*$ 属于 $\text{Span}\{v_1,\dots,v_{i-1}\}$. 我们需要证明 $v_i^{*} \in \text{Span}\{v_1,\dots,v_i\}$. 根据定义 $v_i^*=v_i - \sum_{j=1}^{i-1}\mu_{ij}v_j^*$, 则 $v_i^* \in \text{Span}\{v_1^*,\dots,v_{i-1}^*, v_i\}$. 而 $v_1^*,\dots,v_{i-1}^*$ 属于 $\text{Span}\{v_1,\dots,v_{i-1}\}$，因此 $v_i^* \in \text{Span}\{v_1,\dots,v_{i-1}, v_i\}$，即有 $\supseteq$

