
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
        'dev_wuzhijing', 'f7jBYm9fWqma45Ox7Iv0', '172.16.1.13', '3306', 'zyyx'))


def get_data_from_SQL():
    # 配置连接SQL数据库的信息
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(
        'dev_wuzhijing', 'f7jBYm9fWqma45Ox7Iv0', '172.16.1.13', '3306', 'zyyx'))
    # 提取数据的SQL命令
    sql_query = "SELECT a.report_id, a.stock_code, a.stock_name, b.industry_code, b.industry_name, a.report_type, a.author, a.current_create_date, a.previous_create_date,a.report_year, a.current_forecast_np, a.previous_forecast_np, a.np_adjust_rate, a.entrytime, b.into_date, b.out_date FROM rpt_earnings_adjust a JOIN (SELECT stock_code, industry_code, industry_name, MAX(into_date) as into_date, out_date FROM qt_indus_constituents WHERE industry_level = 1 AND standard_code = 905 GROUP BY stock_code) b ON a.stock_code = b.stock_code WHERE TIMESTAMPDIFF(day, a.current_create_date, a.entrytime) < 5 AND TIMESTAMPDIFF(YEAR, a.current_create_date, a.previous_create_date) < 1 AND YEAR(a.current_create_date) = report_year AND a.current_create_date>='20100101' AND a.current_create_date<='20220831' AND report_type >21 ORDER BY current_create_date ASC "

    # 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
    df = pd.read_sql_query(sql_query, engine)
    return df

df = get_data_from_SQL()

df['Date'] = df['entrytime'].to_numpy().astype('datetime64[M]') #生成月初第一天

#一次去极值
def filter_extreme_3sigma(series,n=3): #3 sigma
    mean = series.mean()
    std = series.std()
    max_range = mean + n*std
    min_range = mean - n*std
    return np.clip(series,min_range,max_range)

df['np_adjust_rate1'] = filter_extreme_3sigma(df['np_adjust_rate'])
df['np_adjust_rate1'].describe()

#把有缺失的股票提出来，可以单独作为一组数据（后面考虑）
df_nan = df[df['np_adjust_rate1'].isnull()]

#缺失值用平均值替代
df['np_adjust_rate1'] = df['np_adjust_rate1'].fillna(df['np_adjust_rate'].mean())

# 获取这一列每个月的第一天
Date=df[['Date']].drop_duplicates(ignore_index=True)


def neutralization(data):
    dummies = pd.get_dummies(data['industry_code'],prefix='industry')
    data = pd.concat([data,dummies],axis=1)
    #OLS回归
    x = data.iloc[:,3:]
    y = data['np_adjust_rate1']
    result = sm.OLS(y,x).fit()
    # print(result.summary())
    data['predict'] = result.predict()
    data['factor'] = data['np_adjust_rate1'] - data['predict'] #获得残差
    return data

def get_group (data_raw,trade): #只能对一行操作，要跑循环使用
    ## 分组进行测试  进行因子检验 IC
    # IC_S = []  # IC 值
    # 选取上一期的持仓
    # 根据因子值进行降序 选取分组1、分组2、分组3、分组4、分组5 股票
    hold_secu = data_raw.sort_values(by='factor1', ascending=False)  # 列索引为行业内码
    # 每组持有的股票数量
    k_num = len(hold_secu) // 10
    hold_group1_sec = hold_secu.iloc[:k_num, : ]
    hold_group2_sec = hold_secu.iloc[k_num:2 * k_num, : ]
    hold_group3_sec = hold_secu.iloc[2 * k_num:3 * k_num, : ]
    hold_group4_sec = hold_secu.iloc[3 * k_num:4 * k_num, : ]
    hold_group5_sec = hold_secu.iloc[4 * k_num:5 * k_num, : ]
    hold_group6_sec = hold_secu.iloc[5 * k_num:6 * k_num, : ]
    hold_group7_sec = hold_secu.iloc[6 * k_num:7 * k_num, : ]
    hold_group8_sec = hold_secu.iloc[7 * k_num:8 * k_num, : ]
    hold_group9_sec = hold_secu.iloc[8 * k_num:9 * k_num, : ]
    hold_group10_sec = hold_secu.iloc[9 * k_num:, : ]

    # 计算IC序列值 下一期股票收益率
    # temp_d = pd.concat([data_raw, secu_pct])
    # temp_ic = [temp_d.T.corr(method='spearman').iloc[0, 1]]
    # 分组持有的行业
    hold_sec_g1, hold_sec_g2, hold_sec_g3, hold_sec_g4, hold_sec_g5, hold_sec_g6, hold_sec_g7, hold_sec_g8, hold_sec_g9, hold_sec_g10 = pd.DataFrame({'stock_code': hold_group1_sec.stock_code.values, 'factor1': hold_group1_sec.factor1.values, 'group': 1}), pd.DataFrame({'stock_code': hold_group2_sec.stock_code.values, 'factor1': hold_group2_sec.factor1.values,'group': 2}), \
                        pd.DataFrame({'stock_code': hold_group3_sec.stock_code.values,'factor1': hold_group3_sec.factor1.values,'group': 3}), pd.DataFrame({'stock_code': hold_group4_sec.stock_code.values,'factor1': hold_group4_sec.factor1.values, 'group': 4}), pd.DataFrame({'stock_code': hold_group5_sec.stock_code.values,'factor1': hold_group5_sec.factor1.values, 'group': 5}),pd.DataFrame({'stock_code': hold_group6_sec.stock_code.values,'factor1': hold_group6_sec.factor1.values, 'group': 6}),\
                        pd.DataFrame({'stock_code': hold_group7_sec.stock_code.values, 'factor1': hold_group7_sec.factor1.values,'group': 7}),pd.DataFrame({'stock_code': hold_group8_sec.stock_code.values, 'factor1': hold_group8_sec.factor1.values,'group': 8}), pd.DataFrame({'stock_code': hold_group9_sec.stock_code.values,'factor1': hold_group9_sec.factor1.values, 'group': 9}),pd.DataFrame({'stock_code': hold_group10_sec.stock_code.values, 'factor1': hold_group10_sec.factor1.values,'group': 10})
    # 数据合并
    hold_sec_gp = pd.concat([hold_sec_g1, hold_sec_g2, hold_sec_g3, hold_sec_g4, hold_sec_g5,hold_sec_g6, hold_sec_g7, hold_sec_g8,hold_sec_g9, hold_sec_g10], ignore_index=True)
    hold_sec_gp.insert(0, 'Trade', trade)  # 日期、内码、分组号

    return hold_sec_gp


#循环获取一个月内所有点评报告的切片
df_final = pd.DataFrame()
for i in range(len(Date)-1):
    trade = Date.Date.iloc[i]
    print("提取月度" + str(trade) + "的数据")
    test = df[df['Date']==Date.Date.iloc[i]]
    # test = test.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    test1 = test.groupby('stock_code')['np_adjust_rate1'].median().reset_index()
    #中性化处理
    #行业与市值提取
    sql_query = ("SELECT a.stock_code, b.industry_code, log(avg(a.mcap)) mcap FROM qt_stk_daily a JOIN (SELECT stock_code, industry_code, industry_name, MAX(into_date) as into_date, out_date FROM qt_indus_constituents WHERE industry_level = 1 AND standard_code = 905 GROUP BY stock_code) b ON a.stock_code = b.stock_code where a.trade_date >= '%s' AND a.trade_date <'%s' GROUP BY a.stock_code") %(trade,Date.Date.iloc[i+1])
    stock_mcap_ind = pd.read_sql_query(sql_query, engine)
    test1 = test1.merge(stock_mcap_ind[['stock_code','industry_code','mcap']],on=['stock_code'],how='inner')

    test1 = neutralization(test1)
    test1.insert(0, 'Trade', trade)

    #每个月盈利预测调整指标分组：
    hold_secu = test1.sort_values(by='factor', ascending=False)  
    hold_sec_gp = get_group(test1,trade)

    df_final = pd.concat([df_final,hold_sec_gp], ignore_index=True)

df_final['factor'].describe()
sns.distplot(df_final['factor'],rug = False, hist=True) #绘制概率密度图

df_final['factor'] = filter_extreme_3sigma(df_final['factor'])

def normalization(series):
    mean=series.mean()
    std=series.std()
    return (series-mean)/std

df_final['factor'] = normalization(df_final['factor'])


#获取全部stocks收益率
sql_query = ("SELECT trade_date, stock_code, change_rate/100 as change_rate FROM qt_stk_daily WHERE trade_date >= '20100201' AND trade_date <= '20220831' ORDER BY trade_date ASC")
stock_pct = pd.read_sql_query(sql_query, engine)
#生成月初信号日期
stock_pct['Date'] = pd.to_datetime(stock_pct['trade_date'])
stock_pct['Date'] = stock_pct['Date'].to_numpy().astype('datetime64[M]')

#计算累计收益率
def get_sec_pct(data):
    # 根据月初生成信号日期、把日期转化为下月的第一天
    data['Date']=pd.to_datetime(data['Trade'])
    # data.index=np.arange(len(data))
    if isinstance(data['Trade'].iloc[0],str):
       data['Date']=[pd.to_datetime(dt)+relativedelta(days=45) for dt in data['Trade']]
    else:
        data['Date'] = [dt + relativedelta(days=45) for dt in data['Trade']]
    data['Date']=data['Date'].to_numpy().astype('datetime64[M]')

    #计算收益率
    data = data.merge(stock_pct[['trade_date','stock_code','change_rate','Date']], on=['stock_code','Date'], how='left')
    
    return data

#得到当期因子值与下一期收益率对照，以便后面计算IC值
df_final1 = get_sec_pct(df_final)

def get_sec_portfolio(data):
    data = data.merge(data.groupby('trade_date')['stock_code'].count().reset_index(), on=['trade_date'], how='left')
    # data['PortfolioRet'] = data['factor_normal']*data['change_rate'] * (1 / data['stock_code_y'])
    data['PortfolioRet'] = data['change_rate'] * (1 / data['stock_code_y'])
    df = data.groupby('trade_date').sum()['PortfolioRet'].to_frame()
    df['portfolio_value'] = np.cumprod(1 + df['PortfolioRet'])

    return df
    

def ret_value(df):
    df1 = df[df['group'] == 1]
    df_1 = get_sec_portfolio(df1)
    df_1.rename(columns={'PortfolioRet': 'PortfolioRet1', 'portfolio_value': 'portfolio_value1'}, inplace=True)

    df2 = df[df['group'] == 2]
    df_2 = get_sec_portfolio(df2)
    df_2.rename(columns={'PortfolioRet': 'PortfolioRet2', 'portfolio_value': 'portfolio_value2'}, inplace=True)

    df3 = df[df['group'] == 3]
    df_3 = get_sec_portfolio(df3)
    df_3.rename(columns={'PortfolioRet': 'PortfolioRet3', 'portfolio_value': 'portfolio_value3'}, inplace=True)

    df4 = df[df['group'] == 4]
    df_4 = get_sec_portfolio(df4)
    df_4.rename(columns={'PortfolioRet': 'PortfolioRet4', 'portfolio_value': 'portfolio_value4'}, inplace=True)

    df5 = df[df['group'] == 5]
    df_5 = get_sec_portfolio(df5)
    df_5.rename(columns={'PortfolioRet': 'PortfolioRet5', 'portfolio_value': 'portfolio_value5'}, inplace=True)

    df6 = df[df['group'] == 6]
    df_6 = get_sec_portfolio(df6)
    df_6.rename(columns={'PortfolioRet': 'PortfolioRet6', 'portfolio_value': 'portfolio_value6'}, inplace=True)

    df7 = df[df['group'] == 7]
    df_7 = get_sec_portfolio(df7)
    df_7.rename(columns={'PortfolioRet': 'PortfolioRet7', 'portfolio_value': 'portfolio_value7'}, inplace=True)

    df8 = df[df['group'] == 8]
    df_8 = get_sec_portfolio(df8)
    df_8.rename(columns={'PortfolioRet': 'PortfolioRet8', 'portfolio_value': 'portfolio_value8'}, inplace=True)

    df9 = df[df['group'] == 9]
    df_9 = get_sec_portfolio(df9)
    df_9.rename(columns={'PortfolioRet': 'PortfolioRet9', 'portfolio_value': 'portfolio_value9'}, inplace=True)

    df10 = df[df['group'] == 10]
    df_10 = get_sec_portfolio(df10)
    df_10.rename(columns={'PortfolioRet': 'PortfolioRet10', 'portfolio_value': 'portfolio_value10'}, inplace=True)

    df_day = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10], axis=1)
    return df_day

#分组计算绝对收益
group_final=ret_value(df_final1)

group_final[['portfolio_value1','portfolio_value2','portfolio_value3','portfolio_value4','portfolio_value5','portfolio_value6','portfolio_value7','portfolio_value8','portfolio_value9','portfolio_value10']].plot(figsize=(16, 9), grid=False, fontsize=20)

plt.rcParams['font.sans-serif'] = ['SimHei'] #正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


#提取指数涨跌幅数据
sql_query = ("SELECT trade_date, index_code, change_rate/100 AS change_rate FROM qt_idx_daily WHERE index_code = '000300' AND trade_date >='20100101' ORDER BY trade_date asc")
index_pct = pd.read_sql_query(sql_query, engine)


def changepct(index_pct,data):

    data = data.merge(index_pct, left_index=True, right_index=True, how='left')

    data['IndustryRet1'] = data['PortfolioRet1'] - data['change_rate']
    data['industry_value1'] = np.cumprod(1 + data['IndustryRet1'])
    data['IndustryRet2'] = data['PortfolioRet2'] - data['change_rate']
    data['industry_value2'] = np.cumprod(1 + data['IndustryRet2'])
    data['IndustryRet3'] = data['PortfolioRet3'] - data['change_rate']
    data['industry_value3'] = np.cumprod(1 + data['IndustryRet3'])
    data['IndustryRet4'] = data['PortfolioRet4'] - data['change_rate']
    data['industry_value4'] = np.cumprod(1 + data['IndustryRet4'])
    data['IndustryRet5'] = data['PortfolioRet5'] - data['change_rate']
    data['industry_value5'] = np.cumprod(1 + data['IndustryRet5'])

    data['IndustryRet6'] = data['PortfolioRet6'] - data['change_rate']
    data['industry_value6'] = np.cumprod(1 + data['IndustryRet6'])
    data['IndustryRet7'] = data['PortfolioRet7'] - data['change_rate']
    data['industry_value7'] = np.cumprod(1 + data['IndustryRet7'])
    data['IndustryRet8'] = data['PortfolioRet8'] - data['change_rate']
    data['industry_value8'] = np.cumprod(1 + data['IndustryRet8'])
    data['IndustryRet9'] = data['PortfolioRet9'] - data['change_rate']
    data['industry_value9'] = np.cumprod(1 + data['IndustryRet9'])
    data['IndustryRet10'] = data['PortfolioRet10'] - data['change_rate']
    data['industry_value10'] = np.cumprod(1 + data['IndustryRet10'])


    return data

#计算分组超额收益
index_pct = index_pct[['trade_date','change_rate']].set_index('trade_date')
group_final1 = changepct(index_pct, group_final)

group_final1[['industry_value1','industry_value2','industry_value3','industry_value4','industry_value5','industry_value6','industry_value7','industry_value8','industry_value9','industry_value10']].plot(figsize=(16, 9), grid=False, fontsize=20)


#计算多空净值
long_short = pd.DataFrame()
long_short['portfolioret'] = group_final['PortfolioRet1'] - group_final['PortfolioRet10']
long_short['portfolio_value'] = np.cumprod(1 + long_short['portfolioret'])
# long_short['industryret'] = group_final1['IndustryRet1'] - group_final1['IndustryRet10']
# long_short['industry_value'] = np.cumprod(1 + long_short['industryret'])

long_short[['portfolio_value']].plot(figsize=(16, 9), grid=False, fontsize=20)

# group_final1.to_excel('/home/wuzhijing/analyst/tables & plots/group_value.xlsx')
# long_short.to_excel('/home/wuzhijing/analyst/tables & plots/long-short.xlsx')

#多空策略评价
result = pd.DataFrame(columns=['多空收益','波动','多空IR','IC均值','IC_IR','IC胜率'],index=['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'])
long_short['year'] = long_short.index.year
group_final1['year'] = group_final1.index.year

result['多头超额'] = np.nan
result['超额波动'] = np.nan
result['夏普比率'] = np.nan
#计算收益、波动、IR（超额与绝对的线是完全一致的，因为基准相抵消了）
for i,year in enumerate(result.index):
    # temp = long_short[long_short.year == int(year)]
    # revenue = (temp.portfolio_value.iloc[-1]/temp.portfolio_value.iloc[0])** (250 / len(temp)) - 1
    # vol = math.sqrt(250)*temp['portfolioret'].std()
    # IR = revenue/vol

    # result['多空收益'].iloc[i] = str(round(revenue * 100, 2)) + '%'
    # result['波动'].iloc[i] = str(round(vol * 100, 2)) + '%'
    # result['多空IR'].iloc[i] = IR
    # result['IC均值'].iloc[i] = str(round(IC_year.IC.iloc[i] *100,2)) + '%'
    # result['IC_IR'].iloc[i] = IC_IR.iloc[i]
    result['IC胜率'].iloc[i] = IC_ratio.sig.iloc[i]/12

result.to_excel('/home/wuzhijing/analyst/tables & plots/result.xlsx')


#计算IC均值
IC = []
for i in range(1,len(Date)):
    trade = Date.Date.iloc[i]
    print("提取月度" + str(trade) + "的数据")
    test = stock_pct[stock_pct['Date']==Date.Date.iloc[i]]
    test1 = test.groupby('stock_code')['change_rate'].mean().reset_index()
    temp = df_final[df_final['Date']==Date.Date.iloc[i]]
    #计算收益率
    temp = temp.merge(test1[['stock_code','change_rate']], on=['stock_code'], how='left')
    temp_ic = temp['factor1'].corr(temp['change_rate'],method='spearman')
    IC.append(temp_ic)

IC = pd.DataFrame(IC)
IC = pd.concat([Date,IC],axis = 1)

IC.index = IC['Date']
IC['year'] = IC.index.year

IC_year = IC.groupby('year')['IC'].mean().reset_index()
IC_std = IC.groupby('year')['IC'].std().reset_index()
IC_IR = IC_year['IC']/IC_std['IC']

#IC胜率
IC['sig'] = np.nan
for i in range(len(IC)):
    if IC.IC.iloc[i]>0:
        IC.sig.iloc[i] = 1
    else:
        IC.sig.iloc[i] = 0
IC_ratio = IC.groupby('year')['sig'].sum().reset_index()

#计算所有数据
#多空
revenue = (long_short.portfolio_value.iloc[-1]/long_short.portfolio_value.iloc[0])** (250 / len(long_short)) - 1
vol = math.sqrt(250)*long_short['portfolioret'].std()
IR = revenue/vol
IC_mean = IC.IC.mean()
IC_std_all = IC.IC.std()
IC_mean/IC_std_all
IC_count = IC.sig.sum()

#纯多头
revenue = (group_final1.industry_value1.iloc[-1]/group_final1.industry_value1.iloc[0])** (250 / len(group_final1)) - 1
vol = math.sqrt(250)*group_final1['IndustryRet1'].std()
IR = revenue/vol

#计算单因子收益曲线
df_final1['factor'] = filter_extreme_3sigma(df_final1['factor'])
df_final1['factor'] = normalization(df_final1['factor'])

# df_final1['factor'].describe()
sns.distplot(df_final1['factor'],rug = False, hist=True) #绘制概率密度图
def get_sec_portfolio1(data):
    data = data.merge(data.groupby('trade_date')['stock_code'].count().reset_index(), on=['trade_date'], how='left')
    data['PortfolioRet'] = data['factor']*data['change_rate'] * (1 / data['stock_code_y'])
    # data['PortfolioRet'] = data['change_rate'] * (1 / data['stock_code_y'])
    df = data.groupby('trade_date').sum()['PortfolioRet'].to_frame()
    df['portfolio_value'] = np.cumprod(1 + df['PortfolioRet'])

    return df

factor_value = get_sec_portfolio1(df_final1)
factor_value[['portfolio_value']].plot(figsize=(16, 9), grid=False, fontsize=20)

#与一致预期指标进行比较
sql_query = ("SELECT stock_code, stock_name, con_year, con_pe, con_peg, con_roe, con_npcgrate_2y, con_npgrate_13w FROM con_forecast_stk")
df_con = pd.read_sql_query(sql_query, engine)


datetimes = pd.to_datetime(df_final['Trade'])
df_final['con_year'] = datetimes.dt.year

df_final1 = df_final.merge(df_con[['stock_code','con_year', 'con_pe', 'con_peg', 'con_roe', 'con_npcgrate_2y', 'con_npgrate_13w']],on=['stock_code','con_year'],how = 'left')


sql_query = ("SELECT stock_code, con_year, AVG(con_pe) AS con_pe, AVG(con_peg) AS con_peg, AVG(con_roe) AS con_roe, AVG(con_npcgrate_2y) AS con_npcgrate_2y, AVG(con_npgrate_13w) AS con_npgrate_13w FROM con_forecast_stk GROUP BY stock_code")
con = pd.read_sql_query(sql_query, engine)
df_final2 = df_final.merge(con[['stock_code','con_year', 'con_pe', 'con_peg', 'con_roe', 'con_npcgrate_2y', 'con_npgrate_13w']],on=['stock_code','con_year'],how = 'left')

df_con_final = pd.DataFrame()
for i in range(len(Date)):
    trade = Date.Date.iloc[i]
    print("提取月度" + str(trade) + "的数据")
    test = df_final1[df_final1['Trade']==Date.Date.iloc[i]]
    test1 = test.groupby('stock_code')['factor','con_pe', 'con_peg', 'con_roe', 'con_npcgrate_2y', 'con_npgrate_13w'].mean().reset_index()
    test1.insert(0, 'Trade', trade)

    df_con_final = pd.concat([df_con_final, test1],ignore_index = True)

#指标归一化
def normalization(series):
    mean=series.mean()
    std=series.std()
    return (series-mean)/std

df_con_final['con_pe'] = normalization(df_con_final['con_pe'])
df_con_final['con_peg'] = normalization(df_con_final['con_peg'])
df_con_final['con_roe'] = normalization(df_con_final['con_roe'])
df_con_final['con_npcgrate_2y'] = normalization(df_con_final['con_npcgrate_2y'])
df_con_final['con_npgrate_13w'] = normalization(df_con_final['con_npgrate_13w'])

# df_con_final.to_excel('/home/wuzhijing/analyst/tables & plots/多指标相关性.xlsx')

#计算秩相关系数
df_con_final['factor'].corr(df_con_final['con_pe'],method='spearman')
df_con_final['factor'].corr(df_con_final['con_peg'],method='spearman')
df_con_final['factor'].corr(df_con_final['con_roe'],method='spearman')
df_con_final['factor'].corr(df_con_final['con_npcgrate_2y'],method='spearman')
df_con_final['factor'].corr(df_con_final['con_npgrate_13w'],method='spearman')
df_con_final['con_pe'].corr(df_con_final['con_peg'],method='spearman')
df_con_final['con_pe'].corr(df_con_final['con_roe'],method='spearman')
df_con_final['con_pe'].corr(df_con_final['con_npcgrate_2y'],method='spearman')
df_con_final['con_pe'].corr(df_con_final['con_npgrate_13w'],method='spearman')
df_con_final['con_peg'].corr(df_con_final['con_roe'],method='spearman')
df_con_final['con_peg'].corr(df_con_final['con_npcgrate_2y'],method='spearman')
df_con_final['con_peg'].corr(df_con_final['con_npgrate_13w'],method='spearman')
df_con_final['con_roe'].corr(df_con_final['con_npcgrate_2y'],method='spearman')
df_con_final['con_roe'].corr(df_con_final['con_npgrate_13w'],method='spearman')
df_con_final['con_npcgrate_2y'].corr(df_con_final['con_npgrate_13w'],method='spearman')

#rec ~ con_npgrate_13w指标绩效
#OLS回归
x = df_con_final['factor']
y = df_con_final['con_npgrate_13w']
result = sm.OLS(y,x).fit()
# print(result.summary())
df_con_final['predict'] = result.predict()
df_con_final['factor1'] = df_con_final['factor'] - df_con_final['predict'] #获得残差

sns.distplot(df_con_final['factor1'],rug = False, hist=True) #绘制概率密度图

df_con_final['stock_code'] = df_con_final['stock_code'].astype(str)
for i in range(len(df_con_final)):
    df_con_final.stock_code.iloc[i] = df_con_final.stock_code.iloc[i].zfill(6)


def filter_extreme_3sigma(series,n=3): #3sigma
    mean = series.mean()
    std = series.std()
    max_range = mean + n*std
    min_range = mean - n*std
    return np.clip(series,min_range,max_range)

#指标归一化
def normalization(series):
    mean=series.mean()
    std=series.std()
    return (series-mean)/std

df_con_final['factor1'] = filter_extreme_3sigma(df_con_final['factor1'])
df_con_final['factor1'] = normalization(df_con_final['factor1'])

df_con_final = pd.read_excel('/home/wuzhijing/analyst/tables & plots/多指标相关性.xlsx', sheet_name='data')

df_final = pd.DataFrame()
for i in range(len(Date)-1):
    trade = Date.Date.iloc[i]
    print("提取月度" + str(trade) + "的数据")
    test = df_con_final[df_con_final['Trade']==Date.Date.iloc[i]]
    
    #每个月盈利预测调整指标分组：
    # hold_secu = test.sort_values(by='factor1', ascending=False)  
    hold_sec_gp = get_group(test,trade)

    df_final = pd.concat([df_final,hold_sec_gp], ignore_index=True)

