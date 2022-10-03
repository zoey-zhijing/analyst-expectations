# "分析师评级上调对应到股价变化"
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from DataAPI import *
import math 
from statsmodels import regression 
import statsmodels.api as sm 
from dateutil.relativedelta import relativedelta
import datetime
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats


engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'dev_name', 'xxxxxxxxxx', 'IP', 'port', 'zyyx'))


def get_data_from_SQL():
    # 配置连接SQL数据库的信息
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'dev_name', 'xxxxxxxxxx', 'IP', 'port', 'zyyx'))
    # 提取数据的SQL命令
    sql_query = "SELECT * FROM rpt_rating_adjust WHERE TIMESTAMPDIFF(day, current_create_date, entrytime) < 5 AND TIMESTAMPDIFF(YEAR, current_create_date, previous_create_date) < 1 AND current_create_date>='20100101' AND current_create_date<='20220831' AND rating_adjust_mark = 2  AND current_gg_rating = 7 ORDER BY current_create_date ASC "

    # 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
    df = pd.read_sql_query(sql_query, engine)
    return df

df = get_data_from_SQL()

df['Date'] = df['entrytime'].to_numpy().astype('datetime64[M]') #生成月初第一天

df_count = df.groupby('Date')['stock_code'].count().reset_index()
# df_count.to_excel('/home/wuzhijing/rating/stock_count.xlsx')

final = pd.DataFrame()
for i in range(len(df)):
    print(i)
    sql_query = ("SELECT trade_date, stock_code, change_rate/100 AS change_rate FROM qt_stk_daily WHERE stock_code = '%s' AND trade_date >= DATE_ADD('%s', interval -30 day) AND trade_date <= DATE_ADD('%s', interval 90 day) ORDER BY trade_date ASC") % (df.stock_code.iloc[i],df.current_create_date.iloc[i],df.current_create_date.iloc[i])
    df_pct = pd.read_sql_query(sql_query, engine)
    df_pct['seq']=np.arange(len(df_pct))

    final = pd.concat([final, df_pct],ignore_index=True)


#计算每日绝对收益
final1 = final.merge(final.groupby('seq')['stock_code'].count().reset_index(), on=['seq'], how='left')
# data['PortfolioRet'] = data['factor_normal']*data['change_rate'] * (1 / data['stock_code_y'])
final1['PortfolioRet'] = final1['change_rate'] * (1 / final1['stock_code_y'])
df_final = final1.groupby('seq').sum()['PortfolioRet'].to_frame()
df_final['portfolio_value'] = np.cumprod(1 + df_final['PortfolioRet'])

df_final.to_excel('/home/wuzhijing/rating/ret_value.xlsx')

#提取指数涨跌幅
sql_query = ("SELECT trade_date, index_code, change_rate/100 AS index_rate FROM qt_idx_daily WHERE index_code = 000300 AND trade_date >= '20091201' AND trade_date <= '20220831' ORDER BY trade_date ASC ")
index_pct = pd.read_sql_query(sql_query,engine)
final2 = final.merge(index_pct[['trade_date','index_rate']],on = ['trade_date'], how = 'left') 
final2['excess_rate'] = final2['change_rate'] - final2['index_rate']

#计算每日超额收益
final3 = final2.merge(final2.groupby('seq')['stock_code'].count().reset_index(), on=['seq'], how='left')
# data['PortfolioRet'] = data['factor_normal']*data['change_rate'] * (1 / data['stock_code_y'])
final3['PortfolioRet'] = final3['excess_rate'] * (1 / final3['stock_code_y'])
df_final1 = final3.groupby('seq').sum()['PortfolioRet'].to_frame()
df_final1['portfolio_value'] = np.cumprod(1 + df_final1['PortfolioRet'])

# df_final1.to_excel('/home/wuzhijing/rating/ret_value1.xlsx')

#评级上调样本行业分布
def get_data_from_SQL():
    # 配置连接SQL数据库的信息
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'dev_wuzhijing', 'f7jBYm9fWqma45Ox7Iv0', '172.16.1.13', '3306', 'zyyx'))
    # 提取数据的SQL命令
    sql_query = "SELECT a.stock_code, a.stock_name, b.industry_code, b.industry_name, a.current_create_date, a.previous_create_date, a.current_gg_rating, a.previous_gg_rating, a.rating_adjust_mark FROM rpt_rating_adjust a JOIN (SELECT stock_code, industry_code, industry_name, MAX(into_date) as into_date, out_date FROM qt_indus_constituents WHERE industry_level = 1 AND standard_code = 905 GROUP BY stock_code) b ON a.stock_code = b.stock_code WHERE TIMESTAMPDIFF(day, a.current_create_date, a.entrytime) < 5 AND TIMESTAMPDIFF(YEAR, a.current_create_date, a.previous_create_date) < 1 AND a.current_create_date>='20100101' AND a.current_create_date<='20220831' AND a.rating_adjust_mark = 2  AND a.current_gg_rating = 7 ORDER BY a.current_create_date ASC "

    # 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
    df = pd.read_sql_query(sql_query, engine)
    return df

df1 = get_data_from_SQL()

df_ind_count = df1.groupby('industry_name')['stock_code'].count().to_frame()
df_ind_count['ind_ratio'] = df_ind_count['stock_code']/16082

df_ind_count.to_excel('/home/wuzhijing/rating/行业分布.xlsx')

plt.rcParams['font.sans-serif'] = ['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.pie(df_ind_count['ind_ratio'],labels=df_ind_count.index)


