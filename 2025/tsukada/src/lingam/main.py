import os,pickle
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import graphviz
import lingam
import semopy
from sklearn import preprocessing
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
    ss = preprocessing.StandardScaler()
    df_std = pd.DataFrame(ss.fit_transform(df_std), columns=df.columns)
    del df

    return df_std

def bootstrap_lingam(n,_df):
    logging.info("start")
    # DirectLiNGAM bootstrap
    n_sampling = n
    model = lingam.DirectLiNGAM()
    model.bootstrap(_df, n_sampling=n_sampling)

    # DAGを保存するなら実行
    save_pickle = os.path.join(output_path,'pickles')
    with open(save_pickle+f'/{n_sampling}_Lingam_bootstrap.pickle', mode='wb') as f:
        pickle.dump(model, f)
    logging.info("Done")

def main():
    logging.info("start")
    df = data_preprocessing()
    bootstrap_lingam(100,df)

if __name__ == '__main__':
    main()


