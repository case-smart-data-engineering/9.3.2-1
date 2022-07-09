from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import ApproxInference
from pgmpy.models import BayesianNetwork

# 近似推理
def approximate_reasoning(bayes_model, cpdsList):
    # 将各个节点条件概率分布添加到BN中
    for cpd in cpdsList:
        bayes_model.add_cpds(cpd)
    # 调用pgmpy库里的近似推理算法
    bayes_infer = ApproxInference(bayes_model)
    return bayes_infer


if __name__ == '__main__':
    # 定义BN模型
    bayes_model = BayesianNetwork([("S", "B"), ("A", "B"), ("B", "L"),
                                   ("B", "C")])

    # 定义条件概率分布
    # variable  节点名称
    # variable_card  节点取值个数
    # values  该节点的概率表
    # evidence 该节点的依赖节点
    # evidence_card   依赖节点的取值个数

    smoking_cpd = TabularCPD(variable="S",
                             variable_card=2,
                             values=[[0.6], [0.4]])

    fever_cpd = TabularCPD(variable="A",
                           variable_card=2,
                           values=[[0.6], [0.4]])

    breath_cpd = TabularCPD(variable="B",
                            variable_card=2,
                            values=[[0.9, 0.83, 0.2, 0.05],
                                    [0.1, 0.17, 0.8, 0.95]],
                            evidence=["S", "A"],
                            evidence_card=[2, 2])

    lung_cpd = TabularCPD(variable="L",
                          variable_card=2,
                          values=[[0.8, 0.1], [0.2, 0.9]],
                          evidence=["B"],
                          evidence_card=[2])

    communicate_cpd = TabularCPD(variable="C",
                                 variable_card=2,
                                 values=[[0.6, 0.01], [0.4, 0.99]],
                                 evidence=["B"],
                                 evidence_card=[2])

    cpdsList = [smoking_cpd, fever_cpd, breath_cpd, lung_cpd, \
        communicate_cpd]

    # 近似推理
    bayes_infer = approximate_reasoning(bayes_model, cpdsList)

    # 查询呼吸困难发生的情况下，病人吸烟的概率
    prob_G = bayes_infer.query(variables=["S"],
                               n_samples=20000,
                               evidence={"L": 1})
    print(prob_G)
