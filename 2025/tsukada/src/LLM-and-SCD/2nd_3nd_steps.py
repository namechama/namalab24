import os
import numpy as np
import pandas as pd
import graphviz
import lingam
from sklearn.preprocessing import StandardScaler
from lingam.utils import print_causal_directions, print_dagc, make_dot, make_prior_knowledge
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import io
from scipy.stats import norm
from copy import deepcopy
from itertools import combinations

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges


print("NumPy",  "ver:", np.__version__)
print("Pandas", "ver:", pd.__version__)
print("Graphviz",   "ver:", graphviz.__version__)
print("LiNGAM", "ver:", lingam.__version__)

np.set_printoptions(precision=3, suppress=True)

# fixing the random seed of np for the repoductivity
np.random.seed(203)

# ログの基本設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rd = "/workspace"
input_path = os.path.join(rd + "/input")
output_path = os.path.join(rd + "/output")

#blank 1
context_X = "健康経営偏差値に関するデータ"

#blank 2
labels_X = ['詳細項目1_1明文化・社内浸透偏差値', '詳細項目1_2情報開示・他社への普及偏差値', '詳細項目2_1経営層の関与偏差値', '詳細項目2_2実施体制偏差値', '詳細項目2_3従業員への浸透偏差値', '詳細項 目3_1目標設定、健診・検診等の活用偏差値', '詳細項目3_2健康経営の実践に向けた土台づくり偏差値', '詳細項目3_3保健指導偏差値', '詳細項目3_4生活習慣の改善偏差値', '詳細項目3_5その他の施策偏差値', '詳細項 目4_1健康診断・ストレスチェック偏差値', '詳細項目4_2労働時間・休職偏差値', '詳細項目4_3課題単位・施 策全体の効果検証・改善偏差値', '課題1健康状態にかかわらず全従業員に対する疾病の発生予防偏差値', '課 題2生活習慣病などの疾病の高リスク者に対する重症化予防偏差値', '課題3メンタルヘルス不調等のストレス関連疾患の発生予防・早期発見・対応偏差値', '課題4従業員の生産性低下防止・事故発生予防偏差値', '課題5女性特有の健康関連課題への対応、女性の健康保持・増進偏差値', '課題6休職後の職場復帰、就業と治療の両立 偏差値', '課題7労働時間の適正化、ワークライフバランス・生活時間の確保偏差値', '課題8従業員間のコミュニケーションの促進偏差値', '課題9従業員の感染症予防（インフルエンザなど）偏差値','総合偏差値']

#blank 4
dataset_explanation_X = "健康への投資を促進することを目的に、企業が従業員の健康管理を経営的な視点で考え、健康の保持・増進につながる取組を戦略的に実践するの健康経営度調査のデータセットである。"


# for PC
dag_est_pc = np.loadtxt(os.path.join(output_path,'total_adj_matrix_pc.csv'), delimiter=',')#loading the csv file of adjacency matrix calculated with PC
prob0_pc_directed = np.loadtxt(os.path.join(output_path,'PC_directed_prob_total.csv'), delimiter=',')#loading the csv file of bootstrap probability matrix for directed edges calculated with PC
prob0_pc_undirected = np.loadtxt(os.path.join(output_path,'PC_undirected_prob_total.csv'), delimiter=',')#loading the csv file of bootstrap probability matrix for undirected edges calculated with PC

# for Exact Search
# dag_est_es = np.loadtxt(os.path.join(output_path,'total_adj_matrix_ES.csv'), delimiter=',')#loading the csv file of adjacency matrix calculated with Exact Search
# prob0_es = np.loadtxt(os.path.join(output_path,'ES_prob_total.csv'), delimiter=',') #loading the csv file of bootstrap probability matrix calculated with ExactSearch


#for DirectLiNGAM
lingam0_adjacency_matrix_ = np.loadtxt(os.path.join(output_path,'total_adj_matrix_LiNGAM.csv'), delimiter=',')#loading the csv file of adjacency matrix calculated with DirectLiNGAM with causal coefficients
prob0_lingam = np.loadtxt(os.path.join(output_path,'LiNGAM_prob_total.csv'), delimiter=',')#loading the csv file of bootstrap probability matrix calculated with DirectLiNGAM


def all_edges_pattern1(adjacency_matrix, labels):
  num_nodes = adjacency_matrix.shape[0]
  text = """All of the edges suggested by the statistical causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} → {labels[i]}
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern1(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    causal_text_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
            else:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]}."

    return causal_text_matrix

def all_edges_pattern2(boot_prob, labels):
  num_nodes = boot_prob.shape[0]
  text = """All of the edges with non-zero bootstrap probabilities suggested by the statistical causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} → {labels[i]} (bootstrap probability = {boot_prob[i,j]})
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern2(boot_prob, labels):
    num_nodes = boot_prob.shape[0]
    causal_text_matrix = np.empty(boot_prob.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
            else:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob[i, j]}."

    return causal_text_matrix

def all_edges_pattern3(adjacency_matrix, labels):
  num_nodes = adjacency_matrix.shape[0]
  text = """All of the edges and their coefficients of the structural causal model suggested by the statistical causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} → {labels[i]} (coefficient = {adjacency_matrix[i,j]})
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern3(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    causal_text_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
            else:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a causal coefficient of {adjacency_matrix[i, j]}."

    return causal_text_matrix

def all_edges_pattern4(adjacency_matrix, boot_prob, labels):
  num_nodes = boot_prob.shape[0]
  text = """All of the edges with non-zero bootstrap probabilities and their coefficients of the structural causal model suggested by the statistical causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} → {labels[i]} (coefficient = {adjacency_matrix[i, j]}, bootstrap probability = {boot_prob[i,j]})
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern4(adjacency_matrix, boot_prob, labels):
    num_nodes = adjacency_matrix.shape[0]
    causal_text_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
            else:
              if adjacency_matrix[i, j] == 0:
                  causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob[i, j]}, but the coefficient is likely to be {adjacency_matrix[i, j]}."

              else:
                  causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob[i, j]}, and the coefficient is likely to be {adjacency_matrix[i, j]}."


    return causal_text_matrix

template_Q1_1 = "We want to carry out causal inference {}, considering {} as variables."
template_Q1_2 = "First, we have conducted the statistical causal discovery with LiNGAM(Linear Non-Gaussian Acyclic Model) algorithm, using a fully standardized dataset on {}."

variables_X = ', '.join(labels_X[:-1]) + ', and ' + labels_X[-1]


Q1_1 = template_Q1_1.format(context_X, variables_X)
Q1_2 = template_Q1_2.format(dataset_explanation_X)

Q1_3 = f"According to the results shown above, it has been determined that"

def create_1st_template_text_matrix(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    causal_1st_template_text_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            causal_1st_template_text_matrix[i, j] = f"""Then, your task is to interpret this result from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.
Please provide an explanation that leverages your expert knowledge on the causal relationship between {labels[j]} and {labels[i]}, and assess the naturalness of this causal discovery result.
Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain."""

    return causal_1st_template_text_matrix

#Pattern 0 is prepared from here.
def create_1st_prompt_matrix_pattern0(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    first_prompt_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 + "\n" + f"""If {labels[j]} is modified, will it have a direct impact on {labels[i]}?
Please provide an explanation that leverages your expert knowledge on the causal relationship between {labels[j]} and {labels[i]}.
Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain."""

    return first_prompt_matrix

#Pattern 1
def create_1st_prompt_matrix_pattern1(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    first_prompt_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    all_edges = all_edges_pattern1(adjacency_matrix, labels)
    causal_texts = create_causal_text_matrix1_pattern1(adjacency_matrix, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(adjacency_matrix, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

#Pattern 2
def create_1st_prompt_matrix_pattern2(boot_prob, labels):
    num_nodes = boot_prob.shape[0]
    first_prompt_matrix = np.empty(boot_prob.shape, dtype=object)

    all_edges = all_edges_pattern3(boot_prob, labels)
    causal_texts = create_causal_text_matrix1_pattern3(boot_prob, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(boot_prob, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix


#Pattern 3
def create_1st_prompt_matrix_pattern3(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    first_prompt_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    all_edges = all_edges_pattern2(adjacency_matrix, labels)
    causal_texts = create_causal_text_matrix1_pattern2(adjacency_matrix, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(adjacency_matrix, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

#Pattern 4
def create_1st_prompt_matrix_pattern4(adjacency_matrix, boot_prob, labels):
    num_nodes = boot_prob.shape[0]
    first_prompt_matrix = np.empty(boot_prob.shape, dtype=object)

    all_edges = all_edges_pattern4(adjacency_matrix, boot_prob, labels)
    causal_texts = create_causal_text_matrix1_pattern4(adjacency_matrix, boot_prob, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(adjacency_matrix, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

# comletion of 1st prompting matrices
first_prompt_matrix_LiNGAM1_X_pattern0 = create_1st_prompt_matrix_pattern0(lingam0_adjacency_matrix_, labels_X)
first_prompt_matrix_LiNGAM1_X_pattern1 = create_1st_prompt_matrix_pattern1(lingam0_adjacency_matrix_, labels_X)
first_prompt_matrix_LiNGAM1_X_pattern2 = create_1st_prompt_matrix_pattern2(prob0_lingam, labels_X)
first_prompt_matrix_LiNGAM1_X_pattern3 = create_1st_prompt_matrix_pattern3(lingam0_adjacency_matrix_, labels_X)
first_prompt_matrix_LiNGAM1_X_pattern4 = create_1st_prompt_matrix_pattern4(lingam0_adjacency_matrix_, prob0_lingam, labels_X)

template_Q1_1 = "We want to carry out causal inference {}, considering {} as variables."
template_Q1_2 = "First, we have conducted the statistical causal discovery with Exact Search algorithm, using a fully standardized dataset on {}."

variables_X = ', '.join(labels_X[:-1]) + ', and ' + labels_X[-1]

Q1_1 = template_Q1_1.format(context_X, variables_X)
Q1_2 = template_Q1_2.format(dataset_explanation_X)

Q1_3 = f"According to the results shown above, it has been determined that"

#Pattern1
def create_1st_prompt_matrix_pattern1(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    first_prompt_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    all_edges = all_edges_pattern1(adjacency_matrix, labels)
    causal_texts = create_causal_text_matrix1_pattern1(adjacency_matrix, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(adjacency_matrix, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

#Pattern2
def create_1st_prompt_matrix_pattern2(boot_prob, labels):
    num_nodes = boot_prob.shape[0]
    first_prompt_matrix = np.empty(boot_prob.shape, dtype=object)

    all_edges = all_edges_pattern3(boot_prob, labels)
    causal_texts = create_causal_text_matrix1_pattern2(boot_prob, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(boot_prob, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

# first_prompt_matrix_ES_X_pattern1 = create_1st_prompt_matrix_pattern1(dag_est_es, labels_X)
# first_prompt_matrix_ES_X_pattern2 = create_1st_prompt_matrix_pattern2(prob0_es, labels_X)

def all_edges_pattern1_PC(adjacency_matrix, labels):
  num_nodes = adjacency_matrix.shape[0]
  text = """All of the directed edges suggested by the statistic causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 1:
              text = text + f"""{labels[j]} → {labels[i]}
"""
  text = text +"""-----
  In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == -1:
              text = text + f"""{labels[j]} － {labels[i]}
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern1_PC(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    causal_text_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if adjacency_matrix[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."
            if adjacency_matrix[i, j] == 1:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]}."
            else:
                causal_text_matrix[i, j] = f"there may be a direct causal relationship between {labels[j]} and {labels[i]}, although the direction has not been determined."
    return causal_text_matrix

def all_edges_pattern2_PC(boot_prob0_directed, boot_prob0_undirected, labels):
  num_nodes = boot_prob0_directed.shape[0]
  text = """All of the directed edges with non-zero bootstrap probabilities suggested by the statistic causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob0_directed[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} → {labels[i]} (bootstrap probability = {boot_prob0_directed[i,j]})
"""
  text = text +"""-----
  In additon to the directed edges above, all of the undirected edges suggested by the statistic causal discovery are below:
-----
"""

  for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if j == i:
                continue
            if boot_prob0_undirected[i, j] == 0:
                continue
            else:
              text = text + f"""{labels[j]} ― {labels[i]} (bootstrap probability = {boot_prob0_undirected[i,j]})
"""
  text = text +"""-----
"""
  return text

def create_causal_text_matrix1_pattern2_PC(boot_prob0_directed, boot_prob0_undirected, labels):
    num_nodes = boot_prob0_directed.shape[0]
    causal_text_matrix = np.empty(boot_prob0_directed.shape, dtype=object)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            if boot_prob0_directed[i, j] == 0 and boot_prob0_undirected[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be no direct impact of a change in {labels[j]} on {labels[i]}."

            if boot_prob0_directed[i, j] != 0 and boot_prob0_undirected[i, j] == 0:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob0_directed[i, j]}."

            if boot_prob0_directed[i, j] == 0 and boot_prob0_undirected[i, j] != 0:
                causal_text_matrix[i, j] = f"there may be a direct causal relationship between {labels[j]} and {labels[i]} with a bootstrap probability of {boot_prob0_undirected[i, j]}, although the direction has not been determined."

            else:
                causal_text_matrix[i, j] = f"there may be a direct impact of a change in {labels[j]} on {labels[i]} with a bootstrap probability of {boot_prob0_directed[i, j]}. In addition, it has also been shown above that there may be a direct causal relationship between {labels[j]} and {labels[i]} with a bootstrap probability of {boot_prob0_undirected[i, j]},although the direction has not completely been determined."

    return causal_text_matrix

template_Q1_1 = "We want to carry out causal inference {}, considering {} as variables."
template_Q1_2 = "First, we have conducted the statistical causal discovery with PC(Peter-Clerk) algorithm, using a fully standardized dataset on {}."

variables_X = ', '.join(labels_X[:-1]) + ', and ' + labels_X[-1]

Q1_1 = template_Q1_1.format(context_X, variables_X)
Q1_2 = template_Q1_2.format(dataset_explanation_X)

Q1_3 = f"According to the results shown above, it has been determined that "#LiNGAMの出力結果のテキストの直前部分。


#Pattern 1
def create_1st_prompt_matrix_pattern1_PC(adjacency_matrix, labels):
    num_nodes = adjacency_matrix.shape[0]
    first_prompt_matrix = np.empty(adjacency_matrix.shape, dtype=object)

    all_edges = all_edges_pattern1_PC(adjacency_matrix, labels)
    causal_texts = create_causal_text_matrix1_pattern1_PC(adjacency_matrix, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(adjacency_matrix, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix

#パターン2
def create_1st_prompt_matrix_pattern2_PC(boot_prob0_directed, boot_prob0_undirected, labels):
    num_nodes = boot_prob0_directed.shape[0]
    first_prompt_matrix = np.empty(boot_prob0_directed.shape, dtype=object)

    all_edges = all_edges_pattern2_PC(boot_prob0_directed, boot_prob0_undirected, labels)
    causal_texts = create_causal_text_matrix1_pattern2_PC(boot_prob0_directed, boot_prob0_undirected, labels)
    causal_1st_template_texts = create_1st_template_text_matrix(boot_prob0_directed, labels)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if j == i:
                continue # 対角成分にも、テキストが入ってしまう場合の例外処理
            first_prompt_matrix[i, j] = Q1_1 +"\n"+ Q1_2 +"\n"+ all_edges + "\n"+ Q1_3 + causal_texts[i, j] +"\n"+ causal_1st_template_texts[i, j]

    return first_prompt_matrix


first_prompt_matrix_PC_X_pattern1 = create_1st_prompt_matrix_pattern1_PC(dag_est_pc, labels_X)
first_prompt_matrix_PC_X_pattern2 = create_1st_prompt_matrix_pattern2_PC(prob0_pc_directed, prob0_pc_undirected)

system_role = "You are a helpful assistant for causal inference."

llm_path = "/workspace/input/llm/gemma-2-2b-jpn-it"
# ローカルに保存したモデルとトークナイザーを読み込む
tokenizer = AutoTokenizer.from_pretrained(llm_path)

# 下記はCPUを使用する場合。
# GPUの場合は、device_map="auto",　torch_dtype=torch.bfloat16　とします。
model = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)

# テキスト生成の例
messages = [
    {"role": "assistant",
             "content": system_role},
    {
        "role": "user",
        "content": causal_2nd_prompt_LiNGAM1_X_pattern4[i,j]
    }
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=3000)
generated_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

print(generated_text.strip())

#pattern 4
for i in range(lingam0_adjacency_matrix_.shape[0]):
    for j in range(lingam0_adjacency_matrix_.shape[0]):
        if i == j:
            continue

        model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="cuda",
                torch_dtype=torch.bfloat16
                )

        messages=[
          {"role": "system",
           "content": system_role},
          {
           "role": "user",
           "content": first_prompt_matrix_LiNGAM1_X_pattern4[i,j]
          }
          ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        outputs = model.generate(inputs, max_new_tokens=3000)

        generated_knowledge_matrix_4_L[i, j]= tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

      print(str(i)+","+str(j))

generated_knowledge_matrix_4_L_df = pd.DataFrame(generated_knowledge_matrix_4_L)
generated_knowledge_matrix_4_L_df.to_csv(os.path.join(output_path,"generated_knowledge_matrix_4_L.csv"), encoding='utf-8')
generated_knowledge_matrix_4_L_df.to_csv(os.path.join(output_path,"generated_knowledge_matrix_4_L_for_excel.csv"), encoding='utf-8-sig')
