from warnings import filterwarnings
import numpy as np
from joblib import load


filterwarnings("ignore")
np.random.seed(42)


def convert_numpy_to_yesi(ary):
    out = {
        "chang": ary[0, 0],
        "zhong": ary[0, 1],
        "duan": ary[0, 2],
        "sui": ary[0, 3],
    }

    return out

def convert_numpy_to_yepian(ary):
    out = {
        "da": ary[0],
        "xiao": ary[1],
        "sui": ary[2],
    }

    return out

class 全流程模型:
    def __init__(self):

        ### 加载模型 ###
        model_dir = "models/"
        # B模块切丝模型
        mdl_切丝_B模块_长丝 = load(model_dir+"mdl_切丝_B模块_长丝.joblib")
        mdl_切丝_B模块_中丝 = load(model_dir+"mdl_切丝_B模块_中丝.joblib")
        mdl_切丝_B模块_短丝 = load(model_dir+"mdl_切丝_B模块_短丝.joblib")
        mdl_切丝_B模块_碎丝 = load(model_dir+"mdl_切丝_B模块_碎丝.joblib")

        # H模块切丝模型
        mdl_切丝_H模块_长丝 = load(model_dir+"mdl_切丝_H模块_长丝.joblib")
        mdl_切丝_H模块_中丝 = load(model_dir+"mdl_切丝_H模块_中丝.joblib")
        mdl_切丝_H模块_短丝 = load(model_dir+"mdl_切丝_H模块_短丝.joblib")
        mdl_切丝_H模块_碎丝 = load(model_dir+"mdl_切丝_H模块_碎丝.joblib")

        # 卷包段吸阻模型
        mdl_吸阻 = load(model_dir+"mdlA_M807_吸阻.joblib")

        ### 常数 ###
        # 膨丝的叶丝结构
        膨丝 = np.array([[26.23, 51.84, 19.85, 2.08]])
        assert np.allclose(膨丝.sum(axis=1), 100)

        # 风选前后叶丝结构的变化
        delta_风选后 = np.array([[-5.32, 1.78, 3.61, -0.07]])
        assert np.allclose(delta_风选后.sum(axis=1), 0)

        # 冷却前后叶丝结构的变化
        delta_冷却后 = np.array([[-1.11, -1.89, 2.98, 0.02]])
        assert np.allclose(delta_冷却后.sum(axis=1), 0)

        # 从落料口到掺配前叶丝结构的变化
        delta_掺配前 = np.array([[-2.215, -0.05, 2.29, -0.025]])
        assert np.allclose(delta_掺配前.sum(axis=1), 0)

        # 掺配前后叶丝结构的变化
        delta_掺配后 = np.array([[-1.33, 1.11, 0.29, -0.07]])
        assert np.allclose(delta_掺配后.sum(axis=1), 0)

        # 从掺配后到储烟丝的变化，第一个元素是整丝率变化，第二个元素是碎丝率变化，基于历史数据总结
        delta_储烟丝 = np.array([[-2.78, 0.36]])

        self.mdl_切丝_B模块_长丝 = mdl_切丝_B模块_长丝
        self.mdl_切丝_B模块_中丝 = mdl_切丝_B模块_中丝
        self.mdl_切丝_B模块_短丝 = mdl_切丝_B模块_短丝
        self.mdl_切丝_B模块_碎丝 = mdl_切丝_B模块_碎丝
        self.mdl_切丝_H模块_长丝 = mdl_切丝_H模块_长丝
        self.mdl_切丝_H模块_中丝 = mdl_切丝_H模块_中丝
        self.mdl_切丝_H模块_短丝 = mdl_切丝_H模块_短丝
        self.mdl_切丝_H模块_碎丝 = mdl_切丝_H模块_碎丝
        self.mdl_吸阻 = mdl_吸阻

        self.膨丝 = 膨丝
        self.delta_风选后 = delta_风选后
        self.delta_冷却后 = delta_冷却后
        self.delta_掺配前 = delta_掺配前
        self.delta_掺配后 = delta_掺配后
        self.delta_储烟丝 = delta_储烟丝

    def predict(self, X_test):
        """
        正向使用：已知所有的自变量，求因变量（成品烟支的吸阻）。
        输入：
            X_test是一个自变量矩阵（9列），行数任意，行数表示样本个数。
        输出：
            吸阻预测值，是一个向量，长度等于X_test的行数。
        """

        # 输入合法性检查
        assert type(X_test) == np.ndarray
        assert X_test.ndim == 2
        assert X_test.shape[1] == 9
        assert np.all(X_test >= 0)
        assert np.all(X_test[:, [0, 1, 2]].sum(axis=1) <= 100)
        assert np.all(X_test[:, [3, 4, 5]].sum(axis=1) <= 100)

        # 建模时各个自变量的范围，如果超出这个范围，预测精度不能保证
        B模块大片率 = X_test[:, 0]
        B模块中片率 = 100 - X_test[:, [0, 1, 2]].sum(axis=1)
        B模块小片率 = X_test[:, 1]
        B模块碎片率 = X_test[:, 2]
        assert np.all((B模块大片率 >= 5) & (B模块大片率 <= 30))
        assert np.all((B模块中片率 >= 30) & (B模块中片率 <= 70))
        assert np.all((B模块小片率 >= 10) & (B模块小片率 <= 40))
        assert np.all((B模块碎片率 >= 0) & (B模块碎片率 <= 16))

        H模块大片率 = X_test[:, 3]
        H模块中片率 = 100 - X_test[:, [3, 4, 5]].sum(axis=1)
        H模块小片率 = X_test[:, 4]
        H模块碎片率 = X_test[:, 5]
        assert np.all((H模块大片率 >= 5) & (H模块大片率 <= 22.5))
        assert np.all((H模块中片率 >= 30) & (H模块中片率 <= 70))
        assert np.all((H模块小片率 >= 15) & (H模块小片率 <= 50))
        assert np.all((H模块碎片率 >= 0) & (H模块碎片率 <= 16))

        填充值 = X_test[:, 6]
        含水率 = X_test[:, 7]
        平衡含水率 = X_test[:, 8]
        assert np.all((填充值 >= 4) & (填充值 <= 5))
        assert np.all((含水率 >= 12) & (含水率 <= 13))
        assert np.all((平衡含水率 >= 12) & (平衡含水率 <= 13))

        # B模块
        片烟结构_B模块 = X_test[:, [0, 1, 2]]
        切丝后_B模块_长丝 = self.mdl_切丝_B模块_长丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_中丝 = self.mdl_切丝_B模块_中丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_短丝 = self.mdl_切丝_B模块_短丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_碎丝 = self.mdl_切丝_B模块_碎丝.predict(片烟结构_B模块 / 100)
        # 进行修正
        切丝后_B模块_总和 = 切丝后_B模块_长丝 + 切丝后_B模块_中丝 + 切丝后_B模块_短丝 + 切丝后_B模块_碎丝
        切丝后_B模块_长丝 = 切丝后_B模块_长丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_中丝 = 切丝后_B模块_中丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_短丝 = 切丝后_B模块_短丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_碎丝 = 切丝后_B模块_碎丝 / 切丝后_B模块_总和 * 100
        self.切丝后_B模块 = np.c_[切丝后_B模块_长丝, 切丝后_B模块_中丝, 切丝后_B模块_短丝, 切丝后_B模块_碎丝]
        self.风选后_B模块 = self.切丝后_B模块 + self.delta_风选后

        # H模块
        片烟结构_H模块 = X_test[:, [3, 4, 5]]
        切丝后_H模块_长丝 = self.mdl_切丝_H模块_长丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_中丝 = self.mdl_切丝_H模块_中丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_短丝 = self.mdl_切丝_H模块_短丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_碎丝 = self.mdl_切丝_H模块_碎丝.predict(片烟结构_H模块 / 100)
        # 进行修正
        切丝后_H模块_总和 = 切丝后_H模块_长丝 + 切丝后_H模块_中丝 + 切丝后_H模块_短丝 + 切丝后_H模块_碎丝
        切丝后_H模块_长丝 = 切丝后_H模块_长丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_中丝 = 切丝后_H模块_中丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_短丝 = 切丝后_H模块_短丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_碎丝 = 切丝后_H模块_碎丝 / 切丝后_H模块_总和 * 100
        self.切丝后_H模块 = np.c_[切丝后_H模块_长丝, 切丝后_H模块_中丝, 切丝后_H模块_短丝, 切丝后_H模块_碎丝]
        self.冷却后_H模块 = self.切丝后_H模块 + self.delta_冷却后

        self.掺配前 = (
            0.63 * self.风选后_B模块 + 0.27 * self.冷却后_H模块 + 0.1 * self.膨丝
        ) + self.delta_掺配前
        self.掺配后 = self.掺配前 + self.delta_掺配后

        self.储烟丝整丝率 = (self.掺配后[:, [0]] + self.掺配后[:, [1]]) + self.delta_储烟丝[:, [0]]
        self.储烟丝碎丝率 = self.掺配后[:, [3]] + self.delta_储烟丝[:, [1]]
        self.储烟丝质量 = np.c_[self.储烟丝整丝率, self.储烟丝碎丝率, X_test[:, [6, 7, 8]]]

        self.吸阻 = self.mdl_吸阻.predict(self.储烟丝质量)
        
        return self.吸阻

    def predict1(self, X_test):
        """
        正向使用：已知所有的自变量，求因变量（成品烟支的吸阻）。
        输入：
            X_test是一个自变量矩阵（9列），行数任意，行数表示样本个数。
        输出：
            吸阻预测值，是一个向量，长度等于X_test的行数。
        """

        # 输入合法性检查
        assert type(X_test) == np.ndarray
        assert X_test.ndim == 2
        assert X_test.shape[1] == 9
        assert np.all(X_test >= 0)
        assert np.all(X_test[:, [0, 1, 2]].sum(axis=1) <= 100)
        assert np.all(X_test[:, [3, 4, 5]].sum(axis=1) <= 100)

        # 建模时各个自变量的范围，如果超出这个范围，预测精度不能保证
        B模块大片率 = X_test[:, 0]
        B模块中片率 = 100 - X_test[:, [0, 1, 2]].sum(axis=1)
        B模块小片率 = X_test[:, 1]
        B模块碎片率 = X_test[:, 2]
        assert np.all((B模块大片率 >= 5) & (B模块大片率 <= 30))
        assert np.all((B模块中片率 >= 30) & (B模块中片率 <= 70))
        assert np.all((B模块小片率 >= 10) & (B模块小片率 <= 40))
        assert np.all((B模块碎片率 >= 0) & (B模块碎片率 <= 16))

        H模块大片率 = X_test[:, 3]
        H模块中片率 = 100 - X_test[:, [3, 4, 5]].sum(axis=1)
        H模块小片率 = X_test[:, 4]
        H模块碎片率 = X_test[:, 5]
        assert np.all((H模块大片率 >= 5) & (H模块大片率 <= 22.5))
        assert np.all((H模块中片率 >= 30) & (H模块中片率 <= 70))
        assert np.all((H模块小片率 >= 15) & (H模块小片率 <= 50))
        assert np.all((H模块碎片率 >= 0) & (H模块碎片率 <= 16))

        填充值 = X_test[:, 6]
        含水率 = X_test[:, 7]
        平衡含水率 = X_test[:, 8]
        assert np.all((填充值 >= 4) & (填充值 <= 5))
        assert np.all((含水率 >= 12) & (含水率 <= 13))
        assert np.all((平衡含水率 >= 12) & (平衡含水率 <= 13))

        # B模块
        片烟结构_B模块 = X_test[:, [0, 1, 2]]
        切丝后_B模块_长丝 = self.mdl_切丝_B模块_长丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_中丝 = self.mdl_切丝_B模块_中丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_短丝 = self.mdl_切丝_B模块_短丝.predict(片烟结构_B模块 / 100)
        切丝后_B模块_碎丝 = self.mdl_切丝_B模块_碎丝.predict(片烟结构_B模块 / 100)
        # 进行修正
        切丝后_B模块_总和 = 切丝后_B模块_长丝 + 切丝后_B模块_中丝 + 切丝后_B模块_短丝 + 切丝后_B模块_碎丝
        切丝后_B模块_长丝 = 切丝后_B模块_长丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_中丝 = 切丝后_B模块_中丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_短丝 = 切丝后_B模块_短丝 / 切丝后_B模块_总和 * 100
        切丝后_B模块_碎丝 = 切丝后_B模块_碎丝 / 切丝后_B模块_总和 * 100
        self.切丝后_B模块 = np.c_[切丝后_B模块_长丝, 切丝后_B模块_中丝, 切丝后_B模块_短丝, 切丝后_B模块_碎丝]
        self.风选后_B模块 = self.切丝后_B模块 + self.delta_风选后

        # H模块
        片烟结构_H模块 = X_test[:, [3, 4, 5]]
        切丝后_H模块_长丝 = self.mdl_切丝_H模块_长丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_中丝 = self.mdl_切丝_H模块_中丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_短丝 = self.mdl_切丝_H模块_短丝.predict(片烟结构_H模块 / 100)
        切丝后_H模块_碎丝 = self.mdl_切丝_H模块_碎丝.predict(片烟结构_H模块 / 100)
        # 进行修正
        切丝后_H模块_总和 = 切丝后_H模块_长丝 + 切丝后_H模块_中丝 + 切丝后_H模块_短丝 + 切丝后_H模块_碎丝
        切丝后_H模块_长丝 = 切丝后_H模块_长丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_中丝 = 切丝后_H模块_中丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_短丝 = 切丝后_H模块_短丝 / 切丝后_H模块_总和 * 100
        切丝后_H模块_碎丝 = 切丝后_H模块_碎丝 / 切丝后_H模块_总和 * 100
        self.切丝后_H模块 = np.c_[切丝后_H模块_长丝, 切丝后_H模块_中丝, 切丝后_H模块_短丝, 切丝后_H模块_碎丝]
        self.冷却后_H模块 = self.切丝后_H模块 + self.delta_冷却后

        self.掺配前 = (
            0.63 * self.风选后_B模块 + 0.27 * self.冷却后_H模块 + 0.1 * self.膨丝
        ) + self.delta_掺配前
        self.掺配后 = self.掺配前 + self.delta_掺配后

        self.储烟丝整丝率 = (self.掺配后[:, [0]] + self.掺配后[:, [1]]) + self.delta_储烟丝[:, [0]]
        self.储烟丝碎丝率 = self.掺配后[:, [3]] + self.delta_储烟丝[:, [1]]
        self.储烟丝质量 = np.c_[self.储烟丝整丝率, self.储烟丝碎丝率, X_test[:, [6, 7, 8]]]

        self.吸阻 = self.mdl_吸阻.predict(self.储烟丝质量)
        xizu = self.吸阻

        result = (
            convert_numpy_to_yesi(self.切丝后_B模块), 
            convert_numpy_to_yesi(self.切丝后_H模块), 
            convert_numpy_to_yesi(self.风选后_B模块), 
            convert_numpy_to_yesi(self.冷却后_H模块), 
            convert_numpy_to_yesi(self.掺配后), 
            xizu[0],
        )    
        return result

    def search(
        self,
        x_old,
        y_target,
        type_search,
        网格_B模块大片率=np.arange(5, 30 + 1e-6, 0.5),
        网格_B模块小片率=np.arange(10, 40 + 1e-6, 0.5),
        网格_B模块碎片率=np.arange(0, 16 + 1e-6, 0.5),
        网格_H模块大片率=np.arange(5, 22.5 + 1e-6, 0.5),
        网格_H模块小片率=np.arange(15, 50 + 1e-6, 0.5),
        网格_H模块碎片率=np.arange(0, 16 + 1e-6, 0.5),
    ):
        """
        反向使用：已知因变量（成品烟支的吸阻）和部分自变量，求其他的自变量。
        反向使用的目的是为了推荐片烟结构。比如，设定好吸阻的目标值，假设H模块的片烟结构保持不变，并且储烟丝的填充值、含水率和平衡含水率保持不变量，那么，为了达到设定好的吸阻目标值，推荐什么样的B模块的片烟结构？
        输入：
            x_old是一个长度为9的自变量向量，表示一个样本。
            y_target是吸阻的目标值，是一个浮点型数字或者整数型数字，即，标量。取值范围1050到1250，这是建模时吸阻的范围，超出这个范围，推荐结果不一定可靠。
            type_search表示推荐的类型，取值为“B”表示只对B模块的片烟结构进行推荐，取值为“H”表示只对H模块的片烟结构进行推荐，取值为“BH”表示对B模块或者H模块的片烟结构进行推荐。只能是这3种取值。
            形如“网格_XXXX”表示搜索的集合，推荐结果只能是集合中的一个元素。集合的格式可以是"np.arange(5, 30 + 1e-6, 0.5)"，表示下限是5，上限是30，步长为0.5，
            一般情况下，使用默认值即可，如果业务需求，可以自己设置，但是，范围不能超出默认范围，比如默认范围是5到30，设置的范围可以是10到25，但是，不能是3到25（下限过小），也不能是5到40（上限过大）。
        输出：
            X_new：新的自变量矩阵，10行9列，比如type_search设置为“B"，那么，X_new的第1到3列为推荐的B模块片烟结构（大片率、小片率和碎片率），其他变量复制x_old对应的部分。给出10个最接近吸阻目标值的推荐方案，因此，X_new有10行。
            y_new：X_new对应的预测值，是一个长度为10的向量。
            y_delta：表示推荐的片烟结构下的吸阻与吸阻目标值之间的差值，即，y_delta = y_new - y_target，是一个长度为10的向量。
            X_new、y_new和y_delta按照y_delta绝对值从小到大排序，即，第一行为最接近吸阻目标值的推荐方案。
        注意：如果y_target设置过大或者过小，那么，可能找不到让吸阻接近y_target的片烟结构，留意y_delta取值，如果绝对值较大，表示找不到，如果绝对值较小，表示能找到。
        """

        # 输入合法性检查
        assert type(y_target) == float or type(y_target) == int
        assert y_target >= 1050 and y_target <= 1250
        assert type_search in ["B", "H", "BH"]

        assert type(x_old) == np.ndarray
        assert x_old.ndim == 1
        assert len(x_old) == 9
        assert np.all(x_old >= 0)
        assert x_old[[0, 1, 2]].sum() <= 100
        assert x_old[[3, 4, 5]].sum() <= 100

        # 建模时各个自变量的范围，如果超出这个范围，预测精度不能保证
        B模块大片率 = x_old[0]
        B模块中片率 = 100 - x_old[[0, 1, 2]].sum()
        B模块小片率 = x_old[1]
        B模块碎片率 = x_old[2]
        assert (B模块大片率 >= 5) and (B模块大片率 <= 30)
        assert (B模块中片率 >= 30) and (B模块中片率 <= 70)
        assert (B模块小片率 >= 10) and (B模块小片率 <= 40)
        assert (B模块碎片率 >= 0) and (B模块碎片率 <= 16)

        H模块大片率 = x_old[3]
        H模块中片率 = 100 - x_old[[3, 4, 5]].sum()
        H模块小片率 = x_old[4]
        H模块碎片率 = x_old[5]
        assert (H模块大片率 >= 5) and (H模块大片率 <= 22.5)
        assert (H模块中片率 >= 30) and (H模块中片率 <= 70)
        assert (H模块小片率 >= 15) and (H模块小片率 <= 50)
        assert (H模块碎片率 >= 0) and (H模块碎片率 <= 16)

        填充值 = x_old[6]
        含水率 = x_old[7]
        平衡含水率 = x_old[8]
        assert (填充值 >= 4) and (填充值 <= 5)
        assert (含水率 >= 12) and (含水率 <= 13)
        assert (平衡含水率 >= 12) and (平衡含水率 <= 13)

        if type_search == "B":
            X_grid = np.array(
                [
                    [
                        B模块大片率,
                        B模块小片率,
                        B模块碎片率,
                        H模块大片率,
                        H模块小片率,
                        H模块碎片率,
                        填充值,
                        含水率,
                        平衡含水率,
                    ]
                    for B模块大片率 in 网格_B模块大片率
                    for B模块小片率 in 网格_B模块小片率
                    for B模块碎片率 in 网格_B模块碎片率
                    if 30 <= (100 - B模块大片率 - B模块小片率 - B模块碎片率) <= 70
                ]
            )

        elif type_search == "H":
            X_grid = np.array(
                [
                    [
                        B模块大片率,
                        B模块小片率,
                        B模块碎片率,
                        H模块大片率,
                        H模块小片率,
                        H模块碎片率,
                        填充值,
                        含水率,
                        平衡含水率,
                    ]
                    for H模块大片率 in 网格_H模块大片率
                    for H模块小片率 in 网格_H模块小片率
                    for H模块碎片率 in 网格_H模块碎片率
                    if 30 <= (100 - H模块大片率 - H模块小片率 - H模块碎片率) <= 70
                ]
            )

        elif type_search == "BH":
            X_grid_B = np.array(
                [
                    [
                        B模块大片率,
                        B模块小片率,
                        B模块碎片率,
                        H模块大片率,
                        H模块小片率,
                        H模块碎片率,
                        填充值,
                        含水率,
                        平衡含水率,
                    ]
                    for B模块大片率 in 网格_B模块大片率
                    for B模块小片率 in 网格_B模块小片率
                    for B模块碎片率 in 网格_B模块碎片率
                    if 30 <= (100 - B模块大片率 - B模块小片率 - B模块碎片率) <= 70
                ]
            )

            X_grid_H = np.array(
                [
                    [
                        B模块大片率,
                        B模块小片率,
                        B模块碎片率,
                        H模块大片率,
                        H模块小片率,
                        H模块碎片率,
                        填充值,
                        含水率,
                        平衡含水率,
                    ]
                    for H模块大片率 in 网格_H模块大片率
                    for H模块小片率 in 网格_H模块小片率
                    for H模块碎片率 in 网格_H模块碎片率
                    if 30 <= (100 - H模块大片率 - H模块小片率 - H模块碎片率) <= 70
                ]
            )

            X_grid = np.r_[X_grid_B, X_grid_H]

        else:
            print("bug!")

        y_grid = self.predict(X_grid)
        y_delta = y_grid - y_target
        idx = np.argsort(np.abs(y_delta))[:10]
        y_delta = y_delta[idx]
        X_new = X_grid[idx]
        y_new = y_grid[idx]

        X_new = X_new[0, :]
        y_new = y_new[0]
        y_delta = y_delta[0]

        result = (
            X_new[6],
            X_new[7],
            X_new[8],
            convert_numpy_to_yepian(X_new[:3]), 
            convert_numpy_to_yepian(X_new[3:6]),
            y_new,
        )   
        return result

"""
正向预测
"""
def mdl_predict(B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv):
    mdl = 全流程模型()
    input = np.array(
        [
            [B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv]
        ]
    )
    return mdl.predict1(input)
    

def main_predict(B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv):
    return mdl_predict(B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv)
    
"""
反向推算
"""
def mdl_search(xizu, B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv, type_search):
    mdl = 全流程模型()
    input = np.array([B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv])
    return mdl.search(input, xizu, type_search)
    

def main_search(xizu, tianchongzhi, hanshuilv, phanshuilv, type_search, B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui):
    return mdl_search(xizu, B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv, type_search)
    

def main1():
    """
    B大片:(5,30)
    B小片:(10,40)
    B碎片:(0,16)
    H大片:(5,22.5)
    H小片:(15,50)
    H碎片:(0,16)
    填充值:(4,5)
    含水率:(12,13)
    平衡含水率:(12,13)
    """
    xizu = 1250 
    X_test = np.array(
        [
            [14.005852, 24.795934, 7.780687, 9.98, 31.23, 6.76, 4.6, 12.6, 12.6],
            [16.005852, 24.795934, 7.780687, 9.98, 31.23, 6.76, 4.6, 12.6, 12.6],
            [14.005852, 24.795934, 7.780687, 11.98, 31.23, 6.76, 4.6, 12.6, 12.6],
        ]
    )
    aa = X_test[3, :]
    B_module_da = aa[0]
    B_module_xiao = aa[1]
    B_module_sui = aa[2]
    H_module_da = aa[3]
    H_module_xiao = aa[4]
    H_module_sui = aa[5]
    tianchongzhi = aa[6]
    hanshuilv = aa[7]
    phanshuilv = aa[8]
    type_search="BH"

    res0 = main_predict(B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv)
    res1 = main_search(xizu, B_module_da, B_module_xiao, B_module_sui, H_module_da, H_module_xiao, H_module_sui, tianchongzhi, hanshuilv, phanshuilv, type_search=type_search)

    aa = 0

if __name__ == '__main__':
    main1()