import numpy as np
import pandas as pd

class RoughSets():
    
    def __init__(self,table):
        
        self.D_col = np.ravel(table.iloc[:,-1])
        self.D = np.array(self.D_col,dtype=np.str_)
        self.C = np.array(table.columns[1:-1],dtype=np.str_)
        self.R = np.array(table.columns,dtype=np.str_)
        self.U = np.matrix(table.iloc[:,:],dtype=np.str_)
        self.Uij = np.ravel(self.U[:,:1])
        self.V = np.unique(np.ravel(self.U[:,1:]))
        self.cores = self.Core(D=[self.R[-1]]
                               ,U=self.U
                               ,R = self.R,C=self.C)
        
    
    def VaRange(self,U=None,R=None):
        """查看变量值域
        
        VaRange(U,R)

        Args:
            U (_type_): _description_
            R (_type_): _description_

        Returns:
            _type_: _description_
        """
        return_eq = {"变量名":[],"值域":[]}
        # 切分出值矩阵
        Vmatrix = U[:,1:]
        # 且分出列明
        Rs = R[1:]

        for R_name,Varray in zip(Rs,Vmatrix):
            # 先将列数据降维到行数据
            lower_dimensional = np.ravel(Varray)
            # 对数据去重得到 Va 也就是属性的值域
            Va = np.unique(lower_dimensional) 
            return_eq["变量名"].append(R_name)
            return_eq["值域"].append(Va)

        return pd.DataFrame(return_eq)
    
    def f(self,x,a,U=None,R=None):
        """信息函数
        f(U,R,a=['天气','气温'],x=["1","2"])

        Args:
            U (_type_): _description_
            R (_type_): _description_
            x (_type_): _description_
            a (_type_): _description_

        Returns:
            _type_: _description_
        """
        a = np.array(a)
        x = np.array(x)
        # 切分出 U 的标签 e_{x}

        U_i = U[:,0]

        # 生成 U 轴的布尔值索引
        m,n = U_i.shape
        x_index = np.zeros(m)
        for x_i in x: 
            x_index += np.ravel(U_i==x_i)
        # 将x所属U的位置数字变为bool值索引 
        x_index = x_index.astype(bool)

        # 生成 R 轴的布尔值索引
        m = R.size
        a_index = np.zeros(m)
        for a_i in a:
            a_index += np.ravel(R == a_i)
        # 将a所属R的位置数字变为bool值索引 
        a_index = a_index.astype(bool)

        # 切分出当前信息
        fx = U[x_index].T[a_index].T
        return pd.DataFrame(data=fx,columns=a,index=x)
    
    def IND(self,A:iter,U=None,R=None,out_dataframe=None):
        """
        给定属性名称，查询是否存在等价类
        print(IND(U,R,A=['气温','天气'],out_dataframe=True))
        print(IND(U,R,A=["湿度"],out_dataframe=True))

        Args:
            U (_type_): _description_
            R (_type_): _description_
            A (iter): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # 切分出 U 的标签 e_{x}
        U_i = U[:,0]
        m = R.size
        A = np.sort(np.array(A,dtype=np.str_))
        a_index = np.zeros(m)
        for a_i in A:
            a_index += np.ravel(R == a_i)
        # 将a所属R的位置数字变为bool值索引 
        a_index = a_index.astype(bool)

        # 切出待比较的列
        U_ij = U.T[a_index].T
        # 去重相同属性
        de_duplicate_Uij = np.unique(U_ij,axis=0)


        A_length = len(A)
        return_eq = {"等价属性列":[],"等价属性":[],"等价类对象":[]}
        
        for de_ in de_duplicate_Uij:
            x_index = np.where(U_ij == de_,True,False).sum(axis=1)==len(A)
            return_eq["等价属性列"].append(A)
            return_eq["等价属性"].append(de_)
            return_eq["等价类对象"].append(np.ravel(U_i[x_index]))
            # print(de_,np.ravel(U_i[x_index]))
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    def isIND(self,A:iter,X:iter,U=None,R=None,out_dataframe=None):
        """
        查询的是否存在等价关系
        isIND(U,R,A=["天气","气温"],X=["1","2"],out_dataframe=True)
        Args:
            U (_type_): _description_
            R (_type_): _description_
            A (iter): _description_
            X (iter): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # U_i = U[:,0]
        # print(U_i)
        step_1 = self.IND(U=U,R=R,A=A)
        
        A = np.sort(np.array(A,dtype=np.str_))
        X = np.sort(np.array(X,dtype=np.str_))

        Equivalence_class_object = step_1["等价类对象"]
        Equivalence_attribute = step_1["等价属性"]
        Equivalencet_attribute_columns = step_1["等价属性列"]
        

        return_eq = {
            "X":[X],"A":[A],
            "等价属性列":[]
            ,"等价属性":[]
            ,"等价类对象":[]
                }
        
        
        for _,x in enumerate(Equivalence_class_object):
            # 求X对象名与等价类中的对象名的交集
            is_intersection = np.intersect1d(X ,x)
            # 如果存在交集，就存储
            if len(is_intersection) >= len(X):
                
                return_eq["等价类对象"].append(Equivalence_class_object[_])
                return_eq["等价属性"].append(Equivalence_attribute[_])
                return_eq["等价属性列"].append(Equivalencet_attribute_columns[_])
        
        
        return_condition = len(return_eq["等价属性列"])>0
        if out_dataframe:
            if return_condition:
                return pd.DataFrame(return_eq)
            return f"查询属性A:{A}，与查询对象X:{X}，没有等价类"
        else:
            if return_condition:
                return return_eq
            return f"查询属性A:{A}，与查询对象X:{X}，没有等价类"
        
        
    def lower_approximation(self,X,A,U=None,R=None,out_dataframe=None):
        """
        下近似
        X_case = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13','14']
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}")
        lower_approximation(U=U,X=X_case,R=R,A=A,out_dataframe=True)
       

        Args:
            X (_type_): _description_
            Robj (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        Robj = self.IND(U=U,R=R,A=A)
        # A = np.sort(np.array(A,dtype=np.str_))
        X = np.sort(np.array(X,dtype=np.str_))

        Equivalence_class_object = Robj["等价类对象"]
        # Equivalence_attribute = Robj["等价属性"]
        # Equivalencet_attribute_columns = Robj["等价属性列"]

        return_eq =[]
        
        for eco in Equivalence_class_object:
            m = len(eco)
            ### 如果eco是X的子集
            is_subsets = np.in1d(eco,X).sum()==m
            if is_subsets:
                for e in eco:
                    return_eq.append(e)
        ### 求所有是X子集的eco集合的并集
        return_eq = {"X":[X],"A":[A],"下近似":[np.unique(return_eq)]}

        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq

    def upper_approximation(self,X,A,U=None,R=None,out_dataframe=None):
        """上近似
        X_case =  ["1","2","3","4"]
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}")
        upper_approximation(U=U,X=X_case,R=R,A=A,out_dataframe=True)

        Args:
            X (_type_): _description_
            Robj (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        Robj = self.IND(U=U,R=R,A=A)
        
        X = np.sort(np.array(X,dtype=np.str_))

        Equivalence_class_object = Robj["等价类对象"]
        # Equivalence_attribute = Robj["等价属性"]
        # Equivalencet_attribute_columns = Robj["等价属性列"]

        return_eq =[]
        
        for eco in Equivalence_class_object:
            ### 求eco与X的交集
            intersection_set = np.intersect1d(eco,X)
            if len(intersection_set)>0:
                ### 将 eco与X的交集插入
                for e in eco:
                    return_eq.append(e)
        ### 求eco与X的交集的并集
        return_eq = {"X":[X],"A":[A]
                    ,"上近似":[
                        np.unique(return_eq)
                    ]}

        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    
    def Pos_A(self,X,A,U=None,R=None,out_dataframe=None):
        """正域
        X_case =  np.ravel(U[:,:1])
        print(f"X:\n{X_case}")
        A=["天气","气温","湿度"]
        print(f"A:\n{A}")
        Pos_A(U=U,X=X_case,R=R,A=A,out_dataframe=True)

        Args:
            U (_type_): _description_
            X (_type_): _description_
            R (_type_): _description_
            A (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        A_lower = self.lower_approximation(U=U,X=X,R=R,A=A)['下近似']
        return_eq = {"X":[X],"A":[A],"正域":A_lower}
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    def NEG_A(self,X,A,U=None,R=None,out_dataframe=None):
        """负域
        X_case =  ["1","2","3","4"]
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}")
        NEG_A(U=U,X=X_case,R=R,A=A,out_dataframe=True)

        Args:
            U (_type_): _description_
            X (_type_): _description_
            R (_type_): _description_
            A (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        A_upper = self.upper_approximation(U=U,X=X,R=R,A=A)["上近似"]
        U_i = np.ravel(U[:,:1])
        # 求U-上近似A-(x)的差集
        NEGx = np.setdiff1d(U_i,A_upper)

        return_eq = {"X":[X],"A":[A],"负域":[NEGx]}
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    def BND_A(self,X,A,U=None,R=None,out_dataframe=None):
        """边界
        
        X_case =  ["1","2","3","4"]
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}")
        BND_A(U=U,X=X_case,R=R,A=A,out_dataframe=True)

        Args:
            U (_type_): _description_
            X (_type_): _description_
            R (_type_): _description_
            A (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        A_upper = self.upper_approximation(U=U,X=X,R=R,A=A)["上近似"]
        A_lower = self.lower_approximation(U=U,X=X,R=R,A=A)['下近似']
        BNGx = np.setdiff1d(A_upper,A_lower)
        
        return_eq = {"X":[X],"A":[A],"边界":[BNGx]}
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    
    def isRoughSet(self,X,A,U=None,R=None,out_dataframe=None):
        """是否是粗糙集
        X_case =  ['4', '10', '14']
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}") 
        print(f"isRoughSet 返回 是否是粗糙集 ：{isRoughSet(U=U,X=X_case,R=R,A=A,out_dataframe=bool)}")
        print(f"isRoughSet 返回 字典数据 ：{isRoughSet(U=U,X=X_case,R=R,A=A,out_dataframe=False)}")
        isRoughSet(U=U,X=X_case,R=R,A=A,out_dataframe=True)


        Args:
            U (_type_): _description_
            X (_type_): _description_
            R (_type_): _description_
            A (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        A_upper = self.upper_approximation(U=U,X=X,R=R,A=A)["上近似"][0]
        A_lower = self.lower_approximation(U=U,X=X,R=R,A=A)['下近似'][0]
        is_equality = set(A_upper) == set(A_lower)
        
        BND_a = self.BND_A(U=U,X=X,R=R,A=A)["边界"]
        BNDisEmptySet = len(BND_a) == 0
        
        return_eq = {
            "X":[X]
            ,"A":[A]
            ,"粗糙/精确集":is_equality & BNDisEmptySet and  "X为A的精确集" or "X为A的粗糙集"
        }
        if out_dataframe and out_dataframe != bool:
            return pd.DataFrame(return_eq)
        elif out_dataframe == bool:
            return is_equality==False

        return return_eq
    
    def Score(self,X,A,U=None,R=None,out_dataframe=True):
        """分类数值标准
        X_case =  ["1","2","3","4"]
        print(f"X:\n{X_case}")
        A=["天气","气温"]
        print(f"A:\n{A}")

        Values(U=U,X=X_case,R=R,A=A,out_dataframe=True)


        Args:
            U (_type_): _description_
            X (_type_): _description_
            R (_type_): _description_
            A (_type_): _description_
            out_dataframe (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        upper_appr = self.upper_approximation(U=U,X=X,R=R,A=A)["上近似"][0]
        lower_appr = self.lower_approximation(U=U,X=X,R=R,A=A)["下近似"][0]
        alpha = len(lower_appr)/len(upper_appr)
        rho = 1-alpha
        gamma = len(lower_appr)/U.shape[0]
        return_eq = {
            "X":[X]
            ,"A":[A]
            ,"近似分类精度":alpha
            ,"粗糙度":rho
            ,"近似分类质量":gamma
        }
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq

    def isRed(self,B:iter,A:iter,U=None,R=None,out_dataframe=None):
        """B是否是A的约简
        A = ["天气"]
        B = ["天气"]
        isRed(U=U,R=R,B=B,A=A)

        Args:
            U (_type_): _description_
            R (_type_): _description_
            B (iter): _description_
            A (iter): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # if U != True:
        #     U = self.U
        # if R != True:
        #     R = self.R

        if set(B).issubset(set(A)):
            Abasic_set = self.IND(U=U,R=R,A=A)["等价类对象"]
            Bbasic_set = self.IND(U=U,R=R,A=B)["等价类对象"]

            if len(Abasic_set) == len(Bbasic_set):
                sorted_Abasic = sorted([sorted([e for e in ab]) for ab in Abasic_set])
                sorted_Bbasic = sorted([sorted([e for e in ab]) for ab in Bbasic_set])
                if sorted_Abasic == sorted_Bbasic:
                    return True
            return False
        return "集合B必须属于集合A的子集"

    def Pos_C(self
              ,D=None
              ,U=None,R=None,C=None,out_dataframe=None):
        """ 求核的用的正域函数
        D = ["类别"]  
        Pos_C(U=U,R=R,D=D,C=C)

        Args:
            U (_type_): _description_
            R (_type_): _description_
            D (_type_): _description_
            C (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # if D != True:
        #     D = [self.R[-1]]
        # if U != True:
        #     U = self.U
        # if R != True:
        #     R = self.R
        # if C != True:
        #     C = self.C
            
        IND_D_ = self.IND(U=U,R=R,A=D)

        PosCn = {}
        IND_D = IND_D_["等价类对象"]
        Dclass = IND_D_['等价属性']
        IND_C = self.IND(U=U,R=R,A=C)["等价类对象"]    
        
        t = -1
        for indd in IND_D:
            t += 1
            for indc in IND_C:
                m = len(indc)
                is_subsets = np.in1d(indc,indd).sum()==m
                if is_subsets:
                    for ui in indc:
                        if t in PosCn:
                            PosCn[t].append(ui)
                        else:
                            PosCn.update({t:[ui]})
                            
        return_eq = {f"{Dclass[No][0]}正域":lower for No,lower in PosCn.items()}
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    
    def lower_app(self,set1,set2):
        """求核用的下近似函数

        Args:
            set1 (_type_): _description_
            set2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        return_eq = {"下近似":[]}
        for s1 in set1:
            for s2 in set2:
                m = len(s2)
                is_subsets = np.in1d(s2,s1).sum()==m
                if is_subsets:
                    for e in s2:
                        return_eq["下近似"].append(e)
        return set(sorted(return_eq["下近似"]))

    def Core(self
             ,D=None
             ,U=None,R=None,C=None,out_dataframe=None):
        """求核

        Args:
            U (_type_): _description_
            R (_type_): _description_
            C (_type_): _description_
            D (_type_): _description_
            out_dataframe (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # if D!=True:
        #     D = [self.R[-1]]
        # if U != True:
        #     U = self.U
        # if R != True:
        #     R = self.R
        # if C != True:
        #     C = self.C
            
        PoscD = [v for k,v in self.Pos_C(U=U,R=R,D=D,C=C).items()]
        return_eq = {"属性名":[],"是否可省略":[],"是否是核":[]}
        Uij = set(np.ravel(U[:,:1]))
        for a in C:
            Ca = list(C)
            Ca.remove(str(a)) 
            INDCai = self.IND(U=U,R=R,A=Ca)["等价类对象"]
            lower_set = self.lower_app(set1=PoscD,set2=INDCai)
            isCore = lower_set != Uij
            return_eq["属性名"].append(a)
            return_eq["是否可省略"].append(isCore!=True)
            return_eq["是否是核"].append(isCore)
        if out_dataframe:
            return pd.DataFrame(return_eq)
        return return_eq
    
    def KnowledgeReduction(self,D,Cn,U,R,C):
        """是否可以同时删除Cn的属性
        D = ["类别"]     
        Cn = ["气温","湿度"]
        print(f"是否可以同时删除{Cn}:{KnowledgeReduction(U=U,R=R,C=C,D=D,Cn=Cn)}")
        Args:
            U (_type_): _description_
            R (_type_): _description_
            C (_type_): _description_
            D (_type_): _description_
            Cn (_type_): _description_

        Returns:
            _type_: bool
            如果回复True就是可以同时删除，如果回复是False就是不可以同时删除
        """
        PoscD = [v for k,v in self.Pos_C(U=U,R=R,D=D,C=C).items()]
        Uij = set(np.ravel(U[:,:1]))
        Ca = list(C)
        for a in Cn:        
            Ca.remove(str(a)) 
        INDCai = self.IND(U=U,R=R,A=Ca,out_dataframe=True)["等价类对象"]
        lower_set = self.lower_app(set1=PoscD,set2=INDCai)
        isCore = lower_set != Uij
        return isCore != True


            
    
if __name__ == '__main__':
    table = pd.DataFrame(
        data = np.matrix(
            [
                [1,"晴",	"热","高","无风","N"],
                [2, '晴', '热', '高', '有风', 'N'],
                [3, '多云', '热', '高', '无风', 'P'],
                [4, '雨', '适中', '高', '无风', 'P'],
                [5, '雨', '冷', '正常', '无风', 'P'],
                [6, '雨', '冷', '正常', '有风', 'N'],
                [7, '多云', '冷', '正常', '有风', 'P'],
                [8, '晴', '适中', '高', '无风', 'N'],
                [9, '晴', '冷', '正常', '无风', 'P'],
                [10, '雨', '适中', '正常', '无风', 'P'],
                [11, '晴', '适中', '正常', '有风', 'P'],
                [12, '多云', '适中', '高', '有风', 'P'],
                [13, '多云', '热', '正常', '无风', 'P'],
                [14, '雨', '适中', '高', '有风', 'N']
                
            ]
        )
        ,columns = ["No.","天气","气温","湿度","风","类别"]
    )
    
    RS = RoughSets(table)
    print(RS.Uij)
    
    print(RS.VaRange(RS.U,RS.R))
    
    print("U:",RS.U)
    print(RS.f(a=['天气','气温'],x=["1","2"],R=RS.R,U=RS.U))
    
    print(RS.IND(A=['气温','天气'],R=RS.R,U=RS.U,out_dataframe=True))
    
    print(RS.IND(A=["湿度"],R=RS.R,U=RS.U,out_dataframe=True))
    
    print(RS.isIND(A=["天气","气温"],X=["1","2"],R=RS.R,U=RS.U,out_dataframe=True))

    X_case = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13','14']
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}")
    print(RS.lower_approximation(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
    
    X_case =  ["1","2","3","4"]
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}")
    print(RS.upper_approximation(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
    
    X_case =  np.ravel(RS.U[:,:1])
    print(f"X:\n{X_case}")
    A=["天气","气温","湿度"]
    print(f"A:\n{A}")
    print(RS.Pos_A(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
    
    X_case =  ["1","2","3","4"]
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}")
    print(RS.NEG_A(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
    
    
    X_case =  ["1","2","3","4"]
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}")
    print(RS.BND_A(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
    
    
    X_case =  ['4', '10', '14']
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}") 
    print(f"isRoughSet 返回 是否是粗糙集 ：{RS.isRoughSet(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=bool)}")
    print(f"isRoughSet 返回 字典数据 ：{RS.isRoughSet(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=False)}")
    print(RS.isRoughSet(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
       
    
    X_case =  ["1","2","3","4"]
    print(f"X:\n{X_case}")
    A=["天气","气温"]
    print(f"A:\n{A}")
    print(RS.Score(U=RS.U,X=X_case,R=RS.R,A=A,out_dataframe=True))
        
        
    A = ["天气"]
    B = ["天气"]
    print(RS.isRed(U=RS.U,R=RS.R,B=B,A=A))
    
    D = ["类别"]  
    print(RS.Pos_C(U=RS.U,R=RS.R,D=D,C=RS.C))
    
    D = ["类别"]        
    print(RS.Core(U=RS.U,R=RS.R,C=RS.C,D=D,out_dataframe=True))
    
    D = ["类别"]     
    Cn = ["气温","湿度"]
    print(f"是否可以同时删除{Cn}:{RS.KnowledgeReduction(U=RS.U,R=RS.R,C=RS.C,D=D,Cn=Cn)}")
    
    print(RS.cores)
    
    Cn = np.array(RS.cores['属性名'])[np.array(RS.cores['是否可省略'])]
    D=[RS.R[-1]]
    print(RS.KnowledgeReduction(D=D,Cn=Cn,U=RS.U,R=RS.R,C=RS.C))