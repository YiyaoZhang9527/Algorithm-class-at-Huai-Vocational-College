scipy线性规划的标准形式如下：

* 求的是 min
  所有的约束为 <= 的形式
  所有的变量均 >=0
  如何变为标准形式？
* 原来是max, 直接乘以 -1求min
  若原来约束为 = ，转为 >= 且 <=（写两个式子，同时成立相当于 = ）
  约束原来为 >= 同样的乘以 -1，就变成了 <=
  若有变量 xi < 0 ，那么用 x1 – x2来替代，其中x1>=0， x2>=0

  [原文链接：](https://blog.csdn.net/weixin_44211968/article/details/123544653)

[scipy.optimize.linprog 实现线性规划](https://www.cnblogs.com/MorStar/p/14967794.html)


[整数规划Python案例](https://blog.csdn.net/abc1234564546/article/details/126263264)


安装cvxpy

pip install cvxpy cvxopt numpy mkl
