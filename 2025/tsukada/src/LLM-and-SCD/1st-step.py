
import os,pickle,logging
import numpy as np
import pandas as pd
import graphviz
import lingam
import semopy
from sklearn.preprocessing import StandardScaler
from lingam.utils import print_causal_directions, print_dagc, make_dot, make_prior_knowledge
import matplotlib.pyplot as plt
import seaborn as sns
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import io
from scipy.stats import norm
from copy import deepcopy
from itertools import combinations
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array, check_scalar

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

print("NumPy",  "ver:", np.__version__)
print("Pandas", "ver:", pd.__version__)
print("Graphviz",   "ver:", graphviz.__version__)
print("LiNGAM", "ver:", lingam.__version__)


np.set_printoptions(precision=3, suppress=True)

# ログの基本設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rd = "/workspace"
input_path = os.path.join(rd + "/input")
output_path = os.path.join(rd + "/output")

# カラムリスト

cols_dict = {
    '詳細項目':[
    "詳細項目1_1明文化・社内浸透偏差値",
    "詳細項目1_2情報開示・他社への普及偏差値",
    "詳細項目2_1経営層の関与偏差値",
    "詳細項目2_2実施体制偏差値",
    "詳細項目2_3従業員への浸透偏差値",
    "詳細項目3_1目標設定、健診・検診等の活用偏差値",
    "詳細項目3_2健康経営の実践に向けた土台づくり偏差値",
    "詳細項目3_3保健指導偏差値",
    "詳細項目3_4生活習慣の改善偏差値",
    "詳細項目3_5その他の施策偏差値",
    "詳細項目4_1健康診断・ストレスチェック偏差値",
    "詳細項目4_2労働時間・休職偏差値",
    "詳細項目4_3課題単位・施策全体の効果検証・改善偏差値"
],
    '健康課題':[
 '課題1健康状態にかかわらず全従業員に対する疾病の発生予防偏差値',
 '課題2生活習慣病などの疾病の高リスク者に対する重症化予防偏差値',
 '課題3メンタルヘルス不調等のストレス関連疾患の発生予防・早期発見・対応偏差値',
 '課題4従業員の生産性低下防止・事故発生予防偏差値',
 '課題5女性特有の健康関連課題への対応、女性の健康保持・増進偏差値',
 '課題6休職後の職場復帰、就業と治療の両立偏差値',
 '課題7労働時間の適正化、ワークライフバランス・生活時間の確保偏差値',
 '課題8従業員間のコミュニケーションの促進偏差値',
 '課題9従業員の感染症予防（インフルエンザなど）偏差値'
    ],
    '認定項目':[
         '認定基準1健康経営の方針等の社内外への発信適否',
 '認定基準2①トップランナーとしての健康経営の普及適否',
 '認定基準3健康づくり責任者の役職適否',
 '認定基準4産業医・保健師の関与適否',
 '認定基準5健保組合等保険者との協議・連携適否',
 '認定基準6健康経営の具体的な推進計画適否',
 '認定基準7②従業員の健康診断の実施（受診率100％）適否',
 '認定基準8③受診勧奨に関する取り組み適否',
 '認定基準9④50人未満の事業場におけるストレスチェックの実施適否',
 '認定基準10⑤管理職・従業員への教育適否',
 '認定基準11⑥適切な働き方の実現に向けた取り組み適否',
 '認定基準12⑦コミュニケ－ションの促進に向けた取り組み適否',
 '認定基準13⑧私病等に関する復職・両立支援の取り組み適否',
 '認定基準14⑨保健指導の実施および特定保健指導実施機会の提供に関する取り組み適否',
 '認定基準15⑩食生活の改善に向けた取り組み適否',
 '認定基準16⑪運動機会の増進に向けた取り組み適否',
 '認定基準17⑫女性の健康保持・増進に向けた取り組み適否',
 '認定基準18⑬長時間労働者への対応に関する取り組み適否',
 '認定基準19⑭メンタルヘルス不調者への対応に関する取り組み適否',
 '認定基準20⑮感染症予防に関する取り組み適否',
 '認定基準21⑯喫煙率低下に向けた取り組み適否',
 '認定基準22受動喫煙対策に関する取り組み適否',
 '認定基準23健康経営の実施についての効果検証適否',
 '認定基準24従業員等の人数が大規模法人部門の人数基準に該当適否',
 '認定基準25回答範囲が法人全体適否',
 '認定基準26回答必須設問（設問文の左に◎が付記されているもの）への回答適否',
 '認定基準27全ての誓約事項を満たしている適否',
 '認定基準28経済産業省による一部回答の公表への同意適否',
 '認定基準適合状況'
    ]


}

def data_preprocessing():
    score_df = pd.read_csv(input_path + "/r3fbsheet_data.csv",header = 3,index_col = None)
    score_df = score_df.rename(columns={'コード': '証券コード'})
    df = score_df[['総合偏差値'] + cols_dict['詳細項目'] + cols_dict['健康課題']]
    del score_df

    # df_sampleを標準化
    df_std = df.copy()
    ss = StandardScaler()
    df_std = pd.DataFrame(ss.fit_transform(df_std), columns=df.columns)
    df

    return df_std,df

#function for evaluating the of model fitting

def evaluate_model_fit(adjacency_matrix, X, is_ordinal=None):
    """ evaluate the given adjacency matrix and return fit indices

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Adjacency matrix representing a causal graph.
        The i-th column and row correspond to the i-th column of X.
    X : array-like, shape (n_samples, n_features)
        Training data.
    is_ordinal : array-like, shape (n_features,)
        Binary list. The i-th element represents that the i-th column of X is ordinal or not.
        0 means not ordinal, otherwise ordinal.

    Return
    ------
    fit_indices : pandas.DataFrame
        Fit indices. This API uses semopy's calc_stats(). See semopy's reference for details.
    """

    # check inputs
    adj = check_array(adjacency_matrix, force_all_finite="allow-nan")
    if adj.ndim != 2 or (adj.shape[0] != adj.shape[1]):
        raise ValueError("adj must be an square matrix.")

    X = check_array(X)
    if X.shape[1] != adj.shape[1]:
        raise ValueError("X.shape[1] and adj.shape[1] must be the same.")

    if is_ordinal is None:
        is_ordinal = np.zeros(X.shape[1])
    else:
        is_ordinal = check_array(is_ordinal, ensure_2d=False).flatten()
    if is_ordinal.shape[0] != adj.shape[1]:
        raise ValueError("is_ordinal.shape[0] and adj.shape[1] must be the same.")

    # build desc
    desc = ""
    eta_names = []

    for i, row in enumerate(adj):
        # exogenous
        if np.sum(np.isnan(row)) == 0 and np.sum(np.isclose(row, 0)) == row.shape[0]:
            continue

        desc += f"x{i:d} ~ "

        for j, elem in enumerate(row):
            if np.isnan(elem):
                eta_name = f"eta_{i}_{j}" if i < j else f"eta_{j}_{i}"
                desc += f"{eta_name} + "
                if eta_name not in eta_names:
                    eta_names.append(eta_name)
            elif not np.isclose(elem, 0):
                desc += f"x{j:d} + "
        desc = desc[:-len(" * ")] + "\n"

    if len(eta_names) > 0:
        desc += "DEFINE(latent) " + " ".join(eta_names) + "\n"

    if sum(is_ordinal) > 0:
        indices = np.argwhere(is_ordinal).flatten()

        desc += "DEFINE(ordinal)"
        for i in indices:
            desc += f" x{i}"
        desc += "\n"

    columns = [f"x{i:d}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=columns)

    m = semopy.Model(desc)
    m.fit(X)

    stats = semopy.calc_stats(m)

    return stats

# adjacency matrix generation from the default output of PC algorithm in causal-learn
def create_adjacency_matrix(cg):

    num_nodes = len(cg.G.nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes),dtype=int)

    for i in range(num_nodes):
        for j in range(num_nodes):
                # i <- j
                if cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == -1:
                    adj_matrix[i, j] = 1
                # i -- j
                elif cg.G.graph[i][j] == -1 and cg.G.graph[j][i] == -1:
                    adj_matrix[i, j] = -1
                # i <->
                elif cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == 1:
                    adj_matrix[i, j] = 2
    return adj_matrix

#function for bootstrap in PC and Exact Search
def bootstrap_PC_edge_probabilities(data, n_sampling):

    num_nodes = data.shape[1]
    directed_edge_counts = np.zeros((num_nodes, num_nodes))
    undirected_edge_counts = np.zeros((num_nodes, num_nodes))
    bidirected_edge_counts = np.zeros((num_nodes, num_nodes))

    for _ in range(n_sampling):

        bootstrap_sample = data
        bootstrap_sample_array = bootstrap_sample.to_numpy()


        cg = pc(bootstrap_sample_array, independence_test_method="fisherz",verbose=False, show_progress=False)


        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == -1:  # i <- j
                        directed_edge_counts[i, j] += 1
                    elif cg.G.graph[i][j] == -1 and cg.G.graph[j][i] == -1:  # i -- j
                        undirected_edge_counts[i, j] += 1
                    elif cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == 1:  # i <-> j
                        bidirected_edge_counts[i, j] += 1


    directed_edge_probabilities = directed_edge_counts / n_sampling
    undirected_edge_probabilities = undirected_edge_counts / n_sampling
    bidirected_edge_probabilities = bidirected_edge_counts / n_sampling

    return directed_edge_probabilities, undirected_edge_probabilities, bidirected_edge_probabilities

def bootstrap_ExactSearch_edge_probabilities(data, n_sampling,super_graph):

    num_nodes = data.shape[1]
    directed_edge_counts = np.zeros((num_nodes, num_nodes))

    for _ in range(n_sampling):

        bootstrap_sample = data
        bootstrap_sample_array = bootstrap_sample.to_numpy()


        dag_est, search_stats = bic_exact_search(bootstrap_sample_array, super_graph=super_graph,verbose=False)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if dag_est[i][j] == 1:  # i <- j
                        directed_edge_counts[i, j] += 1

    directed_edge_probabilities = directed_edge_counts / n_sampling

    return directed_edge_probabilities

def PC_search(n,X):
    # causal discovery
    X_array = X.to_numpy()
    pcg = pc(X_array, independence_test_method="fisherz")
    total_adj_matrix_pc = create_adjacency_matrix(pcg)
    np.savetxt(os.path.join(output_path,'total_adj_matrix_pc.csv'), total_adj_matrix_pc, delimiter=',')


    #evaluation of model fittng stats
    model_stats_pc_total = evaluate_model_fit(total_adj_matrix_pc, X)
    model_stats_pc_total.to_csv(os.path.join(output_path,'model_stats_pc_total.csv'), index=False)

    #bootstrap for 1000 times
    PC_directed_prob_total,PC_undirected_prob_total,PC_bidirected_prob_total = bootstrap_PC_edge_probabilities(X, n_sampling=n)

    np.savetxt(os.path.join(output_path,'PC_directed_prob_total.csv'), PC_directed_prob_total, delimiter=',')
    np.savetxt(os.path.join(output_path,'PC_undirected_prob_total.csv'), PC_undirected_prob_total, delimiter=',')
    np.savetxt(os.path.join(output_path,'PC_bidirected_prob_total.csv'), PC_bidirected_prob_total, delimiter=',')

def DirectLiNGAM(n,X,df):

    # causal discovery
    total_LiNGAM = lingam.DirectLiNGAM(prior_knowledge=None)
    total_LiNGAM.fit(df)
    np.savetxt(os.path.join(output_path,'total_adj_matrix_LiNGAM.csv'), total_LiNGAM.adjacency_matrix_, delimiter=',')

    #evaluation of model fittng stats
    evaluate_model_fit(total_LiNGAM.adjacency_matrix_, X)

      #bootstrap for 1000 times
    bootstrap_LiNGAM_total = total_LiNGAM.bootstrap(X, n_sampling=n)
    LiNGAM_prob_total = bootstrap_LiNGAM_total.get_probabilities(min_causal_effect=0.01)
    np.savetxt(os.path.join(output_path,'LiNGAM_prob_total.csv'), LiNGAM_prob_total, delimiter=',')
    LiNGAM_prob_total

def Exact_Search(n,X):
    # causal discovery
    X_array = X.to_numpy()
    total_adj_matrix_ES, search_stats = bic_exact_search(X_array)
    np.savetxt(os.path.join(output_path,'total_adj_matrix_ES.csv'), total_adj_matrix_ES, delimiter=',')
    #evaluation of model fittng stats
    model_stats_ES_total = evaluate_model_fit(total_adj_matrix_ES, X)
    model_stats_ES_total.to_csv(os.path.join(output_path,'model_stats_ES_total.csv'), index=False)
    #bootstrap for 1000 times
    ES_prob_total = bootstrap_ExactSearch_edge_probabilities(X, n_sampling=n,super_graph=None)

    np.savetxt(os.path.join(output_path,'ES_prob_total.csv'), ES_prob_total, delimiter=',')


def main():
    logging.info("start")
    _X,_df = data_preprocessing()
    logging.info("PC")
    PC_search(1000,_X)
    logging.info("DirectLiNGAM")
    DirectLiNGAM(1000,_X,_df)
    # logging.info("ES")
    # Exact_Search(1,_X)
    logging.info("end")

main()


