# 导入类库

import os.path
import os
import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from pandas import read_csv
import re
import pandas as pd
import numpy as np
import itertools
import sys



def get_dict(file):                                           #该函数可以获得蛋白质ID与其序列对应的字典或
    if file=='df_molecule.csv':
        fig_dict=pd.read_csv(file)[['Molecule_ID','Fingerprint']].T.to_dict('series')
    elif file=='df_protein_train.csv' or file=='df_protein_test.csv' :
        pro=open(file,'r').read().upper()                     #将蛋白质序列文件中的小写改为大写字母
        pro_out=open(file,'w')
        pro_out.write(pro)
        pro_out.close()
        fig_dict=pd.read_csv(file)[['PROTEIN_ID','SEQUENCE']].T.to_dict('series')
    else:
        print('文件格式错误')
        sys.exit()
    return fig_dict


def get_new_pro(id_pro, pramate_file):                        #该函数可以获得蛋白质序列进行数字化处理后的矩阵
    pro_result={}
    for key,valuex in id_pro.items():
        value=list(valuex)[-1]
        length=len(value)
        pro_mol={'G':75.07,'A':89.09,'V':117.15,'L':131.17,'I':131.17,'F':165.19,'W':204.23,'Y':181.19,'D':133.10,'N':132.12,'E':147.13,'K':146.19,'Q':146.15,'M':149.21,'S':105.09,'T':119.12,'C':121.16,'P':115.13,'H':155.16,'R':174.20}
        pramate_file_dict = pd.read_csv(pramate_file, index_col='aa').T.to_dict('series')
        pro_n_8_maxitic=np.array([pramate_file_dict[value[0]],pramate_file_dict[value[1]]])
        pro_line=np.array([pro_mol[value[0]],pro_mol[value[1]]])
        for i in value[2:]:
            pro_n_8_maxitic=np.row_stack((pro_n_8_maxitic,pramate_file_dict[i])) #得到n*属性 的计算矩阵
            pro_line= np.append(pro_line,pro_mol[i])
        Lag=list(np.dot(pro_line,pro_n_8_maxitic)/float(length))
        Lag=[ str(i) for i in Lag ]
        pro_result[str(key)] =str(key)+','+','.join(Lag)
    return pro_result

def get_AC_figuer(file_fig_dict):                             #该函数可以获得分子指纹进行数字化处理后的矩阵
    fig = []
    for i in itertools.product('01', repeat=8):
        fig.append(''.join(list(i)))
    out={}
    for k, vx in file_fig_dict.items():
        fig_nu_dict = {}
        v=''.join([ str(i) for i in list(vx)[1:] ]).replace(', ','')
        s = 0
        e = 8
        for ii in range(len(v) - 7):
            read = v[s:e]
            if read in fig_nu_dict:
                fig_nu_dict[read] = fig_nu_dict[read] + 1
            else:
                fig_nu_dict[read] = 1
            s = s + 1
            e = e + 1
        fig_list=[]
        for i in fig:
            if i in fig_nu_dict:
                fig_list.append(str(fig_nu_dict[i]))
            else:
                fig_list.append('0')
        out[str(k)]=str(k)+','+','.join(fig_list)
    return out

def merge_file(new_fig,new_pro,pro_mol_id_file,out_file):     #该函数将蛋白质序列数字矩阵，分子指纹矩阵，小分子18个属性进行融合
    df=pd.read_csv(pro_mol_id_file)
    new_pro=pd.read_csv('new_pro.list',sep='\t')
    new_fig=pd.read_csv('new_fig.list',sep='\t')
    nu_18=pd.read_csv('df_molecule.csv')[['Molecule_ID','cyp_3a4','cyp_2c9','cyp_2d6','ames_toxicity','fathead_minnow_toxicity','tetrahymena_pyriformis_toxicity','honey_bee','cell_permeability','logP','renal_organic_cation_transporter','CLtotal','hia','biodegradation','Vdd','p_glycoprotein_inhibition','NOAEL','solubility','bbb']]
    df['Protein_ID']=df['Protein_ID'].astype(int)
    result=pd.merge(new_pro,df,on='Protein_ID')
    result=pd.merge(new_fig, result, on='Molecule_ID')
    result=pd.merge(nu_18, result, on='Molecule_ID')
    del result['Molecule_ID']
    del result['Protein_ID']
    result.to_csv(out_file,header=True,index=False)

def pro_mol_result(df_protein,df_molecule,df_affinity,df_out): #该函数调用其它函数生成最后的分析矩阵
    new_fig=pd.DataFrame([get_AC_figuer(get_dict(df_molecule))]).T[0].str.split(',', expand=True)
    new_fig.columns = ['Molecule_ID'] + ['Molecule_%s'%i for i in range(256)]
    new_fig.to_csv('new_fig.list',sep='\t',index=False)
    new_pro=pd.DataFrame([get_new_pro(get_dict(df_protein),'aa2.csv')]).T[0].str.split(',', expand=True)
    new_pro.columns = ['Protein_ID'] + ['Protein_%s'%i for i in range(14)]
    new_pro.to_csv('new_pro.list',sep='\t',index=False)
    merge_file(new_fig,new_pro,df_affinity,df_out)
    os.remove('new_fig.list')
    os.remove('new_pro.list')


"蛋白质参数14，单次最好成绩1.31"
def protein_14():
    print('-------------------------------------蛋白质参数14，单次最好成绩1.31-----------------------------------------------------')
    dataset = pd.read_csv('df_train.csv')
    result = pd.read_csv('df_test.csv')

    # 设置参数数量
    NUM = 288

    array = dataset.values
    X = array[:, 0:NUM]  # 总共356个参数需要导入进去
    Y = array[:, NUM]  # 第357列是ki值
    validation_size = 0.2  # 数据八二分
    seed = 7

    # 分离线上线下测试数据
    X_model, X_pred, Y_model, Y_pred = train_test_split(X, Y, test_size=validation_size,
                                                        random_state=seed)  # 建立模型model  线下验证pred

    print('这是线上分离自验证数据...')
    print('模型集', X_model.shape)
    print('线上验证集 ', X_pred.shape)

    # 确认因变量值
    print(Y)


    train = lgb.Dataset(X_model, label=Y_model)
    valid = train.create_valid(X_pred, label=Y_pred)

    for i in range(5):
        params = {

            'boosting_type': 'gbdt',

            'objective': 'rmse',

            'metric': 'rmse',

            'min_child_weight': 3,

            'num_leaves': 20,

            'lambda_l2': 10,

            'subsample': 0.7,

            'colsample_bytree': 0.7,

            'learning_rate': 0.05,

            'seed': 2017,

            'nthread': 12,

            'bagging_fraction': 0.7,

            'bagging_freq': 100,



        }

        starttime = datetime.datetime.now()
        num_round = 40000
        gbm = lgb.train(params,

                        train,

                        num_round,

                        verbose_eval=500,

                        valid_sets=[train, valid],

                        early_stopping_rounds=200

                        )
        endtime = datetime.datetime.now()
        print((endtime - starttime))

        nowTime = datetime.datetime.now().strftime('%m%d%H%M')  # 现在
        name = 'result/' + 'lgb_' + nowTime + '-' + str(num_round) + '.csv'
        result = pd.read_csv('df_test.csv')

        pred = gbm.predict(result.values[:, :NUM])
        result = read_csv('dataset/df_affinity_test_toBePredicted.csv')
        result['Ki'] = pred
        result.to_csv(name, index=False)

        print(result.head(10))

    print('输出完成...')

"词向量处理方式,单次训练结果上交最好1.29"
def wordvec_way():
    print('-------------------------------------词向量处理方式,单次训练结果上交最好1.29-----------------------------------------------------')

    # 数据读取
    df_protein_train = pd.read_csv('dataset/df_protein_train.csv')  # 1653
    df_protein_test = pd.read_csv('dataset/df_protein_test.csv')  # 414
    protein_concat = pd.concat([df_protein_train, df_protein_test])
    df_molecule = pd.read_csv('dataset/df_molecule.csv')  # 111216
    df_affinity_train = pd.read_csv('dataset/df_affinity_train.csv')  # 165084
    df_affinity_test = pd.read_csv('dataset/df_affinity_test_toBePredicted.csv')  # 41383
    df_affinity_test['Ki'] = -11
    data = pd.concat([df_affinity_train, df_affinity_test])


    # 1、Fingerprint分子指纹处理展开
    feat = []
    for i in range(0, len(df_molecule)):
        feat.append(df_molecule['Fingerprint'][i].split(','))
    feat = pd.DataFrame(feat)
    feat = feat.astype('int')

    feat.columns = ["Fingerprint_{0}".format(i) for i in range(0, 167)]
    feat["Molecule_ID"] = df_molecule['Molecule_ID']
    data = data.merge(feat, on='Molecule_ID', how='left')

    # 2、df_molecule其他特征处理
    feat = df_molecule.drop('Fingerprint', axis=1)
    data = data.merge(feat, on='Molecule_ID', how='left')

    # 3、protein 蛋白质 词向量训练
    n = 128
    texts = [[word for word in re.findall(r'.{3}', document)]
             for document in list(protein_concat['Sequence'])]

    model = Word2Vec(texts, size=n, window=4, min_count=1, negative=3,
                     sg=1, sample=0.001, hs=1, workers=4)

    vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])
    vectors['Word'] = list(model.wv.vocab)
    vectors.columns = ["vec_{0}".format(i) for i in range(0, n)] + ["Word"]

    wide_vec = pd.DataFrame()
    result1 = []
    aa = list(protein_concat['Protein_ID'])
    for i in range(len(texts)):
        result2 = []
        for w in range(len(texts[i])):
            result2.append(aa[i])
        result1.extend(result2)
    wide_vec['Id'] = result1

    result1 = []
    for i in range(len(texts)):
        result2 = []
        for w in range(len(texts[i])):
            result2.append(texts[i][w])
        result1.extend(result2)
    wide_vec['Word'] = result1

    del result1, result2

    wide_vec = wide_vec.merge(vectors, on='Word', how='left')
    wide_vec = wide_vec.drop('Word', axis=1)
    wide_vec.columns = ['Protein_ID'] + ["vec_{0}".format(i) for i in range(0, n)]

    del vectors

    name = ["vec_{0}".format(i) for i in range(0, n)]

    feat = pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()
    feat.columns = ["Protein_ID"] + ["mean_ci_{0}".format(i) for i in range(0, n)]
    data = data.merge(feat, on='Protein_ID', how='left')

    #################################### lgb ############################

    train_feat = data[data['Ki'] > -11].fillna(0)
    testt_feat = data[data['Ki'] <= -11].fillna(0)
    label_x = train_feat['Ki']
    label_y = testt_feat['Ki']

    submission = testt_feat[['Protein_ID', 'Molecule_ID']]
    len(testt_feat)
    train_feat = train_feat.drop('Ki', axis=1)
    testt_feat = testt_feat.drop('Ki', axis=1)
    train_feat = train_feat.drop('Protein_ID', axis=1)
    testt_feat = testt_feat.drop('Protein_ID', axis=1)
    train_feat = train_feat.drop('Molecule_ID', axis=1)
    testt_feat = testt_feat.drop('Molecule_ID', axis=1)

    # lgb算法
    train = lgb.Dataset(train_feat, label=label_x)
    test = lgb.Dataset(testt_feat, label=label_y, reference=train)
    for i in range(5):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l2',
            'metric': 'l2',
            'min_child_weight': 3,
            'num_leaves': 2 ** 5,
            'lambda_l2': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'learning_rate': 0.05,
            'seed': 1600,
            'nthread': 12,
            'bagging_fraction': 0.8,
            'bagging_freq': 100,


        }

        num_round = 25000
        gbm = lgb.train(params,
                        train,
                        num_round,
                        verbose_eval=200,
                        valid_sets=[train, test]
                        )

        preds_sub = gbm.predict(testt_feat)

        # 结果保存
        nowTime = datetime.datetime.now().strftime('%m%d%H%M')  # 现在
        name = 'result/lgb_' + nowTime + '.csv'
        submission['Ki'] = preds_sub
        submission.to_csv(name, index=False)

"将两种模型求平均，单词最好1.25"
def result_analyze():
    print('-------------------------------------将两种模型求平均，单词最好1.25-----------------------------------------------------')

    file_dir = 'result'
    for root, dirs, files in os.walk(file_dir):
        print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        print(files)  # 当前路径下所有非目录子文件
    sum = np.zeros(41383)
    for i in files:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == '.csv':
            print(i);
            result = pd.read_csv('result/' + i)
            # print(result.head(10))
            sum = result['Ki'].values + sum
    upload = sum / len(files)


    result = read_csv('dataset/df_affinity_test_toBePredicted.csv')
    result['Ki'] = upload
    print(result.head(10))
    name = 'result/result_upload_finally'  + '.csv'  # 现在
    result.to_csv(name, index=False)


if __name__ == "__main__":
    pro_mol_result('df_protein_train.csv', 'df_molecule.csv', 'df_affinity_train.csv', 'df_train.csv')
    pro_mol_result('df_protein_test.csv', 'df_molecule.csv', 'df_affinity_test_toBePredicted.csv', 'df_test.csv')

    protein_14()

    wordvec_way()

    result_analyze()

