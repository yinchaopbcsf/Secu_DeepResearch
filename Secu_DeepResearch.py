import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
import copy
import numpy as np
# 处理包含型K线
def Cut_KStick(x):
    x["dt_h"] = x["high"].diff(1)
    x["dt_l"] = x["low"].diff(1)
    x["Position_Code"] = None
    id4 = x["Position_Code"][(x["dt_l"] <= 0) & (x["dt_h"] >= 0)].index
    id3 = x["Position_Code"][(x["dt_l"] >= 0) & (x["dt_h"] <= 0)].index
    id2 = x["Position_Code"][(x["dt_l"] <= 0) & (x["dt_h"] <= 0)].index
    id1 = x["Position_Code"][(x["dt_l"] >= 0) & (x["dt_h"] >= 0)].index

    x.loc[id1, "Position_Code"] = 1
    x.loc[id2, "Position_Code"] = 2
    x.loc[id3, "Position_Code"] = 3
    x.loc[id4, "Position_Code"] = 4

    for i in range(2, len(x)):
        if (x["Position_Code"][i] == 4 and x["Position_Code"][i - 1] == 1 and x["Position_Code"][i - 2] == 2):
            x.iloc[i - 1, 2] = x.iloc[i, 2]
            x.iloc[i, 8] = 5
        elif (x["Position_Code"][i] == 3 and x["Position_Code"][i - 1] == 1 and x["Position_Code"][i - 2] == 2):
            x.iloc[i - 1, 3] = x.iloc[i, 3]
            x.iloc[i, 8] = 5
        elif (x["Position_Code"][i] == 3 and x["Position_Code"][i - 1] == 2 and x["Position_Code"][i - 2] == 1):
            x.iloc[i - 1, 2] = x.iloc[i, 2]
            x.iloc[i, 8] = 6
        elif (x["Position_Code"][i] == 4 and x["Position_Code"][i - 1] == 2 and x["Position_Code"][i - 2] == 1):
            x.iloc[i - 1, 3] = x.iloc[i, 3]
            x.iloc[i, 8] = 6
        elif (x["Position_Code"][i] == 3 and x["Position_Code"][i - 1] == 1 and x["Position_Code"][i - 2] == 1):
            x.iloc[i - 1, 3] = x.iloc[i, 3]
            x.iloc[i, 8] = 7
        elif (x["Position_Code"][i] == 4 and x["Position_Code"][i - 1] == 1 and x["Position_Code"][i - 2] == 1):
            x.iloc[i - 1, 2] = x.iloc[i, 2]
            x.iloc[i, 8] = 7
        elif (x["Position_Code"][i] == 3 and x["Position_Code"][i - 1] == 2 and x["Position_Code"][
            i - 2] == 2):
            x.iloc[i - 1, 2] = x.iloc[i, 2]
            x.iloc[i, 8] = 8
        elif (x["Position_Code"][i] == 4 and x["Position_Code"][i - 1] == 2 and x["Position_Code"][i - 2] == 2):
            x.iloc[i - 1, 3] = x.iloc[i, 3]
            x.iloc[i, 8] = 8
    id_drop = x[(x["Position_Code"] == 5) | (x["Position_Code"] == 6) | (x["Position_Code"] == 7) | (
            x["Position_Code"] == 8)].index
    x.drop(id_drop, inplace=True)
# 根据价格判断并处理分型
def Shape_Top_or_Bottom(x):
    # 处理分型
    x["Top_Bottom"] = 0
    # 判断各点分型类别
    for i in range(len(x)):
        if (x["Position_Code"][i - 1] == 1 and x["Position_Code"][i] == 2):
            x.iloc[i - 1, 9] = "Top"
        elif (x["Position_Code"][i - 1] == 2 and x["Position_Code"][i] == 1):
            x.iloc[i - 1, 9] = "Bottom"
    # 清洗分型序列为标准化缠论形式
    c_v = [0, 0]
    t_v = [0, 0]
    for i in range(len(x)):
        if (x.iloc[i, 9] == "Top" or x.iloc[i, 9] == "Bottom") and (t_v[1] == 0):
            if (x.iloc[i+1, 9] == "Top" or x.iloc[i+1, 9] == "Bottom") and (x.iloc[i, 9] != x.iloc[i+1, 9]):
                x.iloc[i, 9] = 0
                x.iloc[i+1, 9] = 0
            else:
                t_v[0] = x.iloc[i, 9]
                t_v[1] = i
        elif (x.iloc[i, 9] == "Top" or x.iloc[i, 9] == "Bottom") and (t_v[1] > 0):
            c_v[0] = x.iloc[i, 9]
            c_v[1] = i
            # 同一分型
            if c_v[0] == t_v[0]:
                x.iloc[t_v[1], 9] = 0
                t_v[0] = x.iloc[i, 9]
                t_v[1] = i
            # 不同分型
            elif c_v[0] != t_v[0]:
                # 完全相邻K线、仅隔1根K线
                if i - t_v[1] <= 2:
                    x.iloc[i, 9] = 0
                    x.iloc[i - 1, 9] = 0
                # 不相邻K线
                elif i - t_v[1] > 2:
                    t_v[0] = x.iloc[i, 9]
                    t_v[1] = i
# 提取缠论顶底分型数据
def extractPrice(x):
    hid = x[x["Top_Bottom"] == "Top"].index
    lid = x[x["Top_Bottom"] == "Bottom"].index
    hdf = pd.DataFrame(x.loc[hid,"high"],index = hid)
    hdf["type"] = "Top"
    hdf.columns = ["P", "type"]
    ldf = pd.DataFrame(x.loc[lid,"low"],index = lid)
    ldf["type"] = "Bottom"
    ldf.columns = ["P", "type"]
    df = hdf.append(ldf)
    return df.sort_index()
# 计算同一序列的同方向数值的累计和函数,生成一个Series
def Drt_CalSum(x):
    drt = 1
    s = 0
    j = 0
    t = list()
    for i in range(len(x)):
        if x[i] * drt >= 0:
            s = s + x[i]
        elif x[i] * drt < 0:
            drt = (-1) * drt
            t.append(s)
            s = 0
            j = j + 1
    return pd.Series(t)
# 画缠论折线图
def plotConciseLine(x,y,nm):
    cht_fig = plt.figure(figsize=(10,6.18),dpi= 666)
    cht = cht_fig.add_subplot(1,1,1)
    cht.plot(x.index,x["close"])
    cht.plot(y.index,y["P"])
    cht_fig.savefig("{}.png".format(nm))
#计算同一序列的同方向数值的累计和函数,生成一个Series
def Drt_CalSum(x):
    drt = 1
    s = 0
    t = list()
    for i in range(len(x)):
        if x.iloc[i] * drt >= 0:
            s = s + x.iloc[i]
        elif x[i] * drt < 0:
            drt = (-1) * drt
            t.append(s)
            s = 0
    if len(t) == 0:
        print("本序列始终为同一符号，其合计数为{}".format(round(s,3)))
        return s
    else:
        return pd.Series(t)
# 求Series无极端值的平均数
def noExtrameMean_Series(x):
    if isinstance(x,pd.Series):
        try:
            tep_v = list()
            ratio_abandon = np.linspace(0, 0.49, 100)
            for i in ratio_abandon:
                z = x.iloc[int(round(x.__len__() * i)):int(round(x.__len__() * (1 - i)))]
                tep_v.append(z.mean())
            tep_v = np.array(tep_v)
            return tep_v.mean()
        except:
            return None
    else:
        print("必须使用pd.Series数据类型！")
# 求dataframe无极端值的平均数
def noExtrameMean_DataFrame(x):
    if isinstance(x,pd.DataFrame):
        try:
            return x.apply(noExtrameMean_Series)
        except:
            return None
    else:
        print("必须使用pd.DataFrame数据类型！")
# 证券研究分析类变量
class Stock_Research():
    def __init__(self,code):
        self.code = code
        # 取财报三表全量数据及价格数据
        self.original_price = ts.get_k_data(self.code, ktype="M")
        pf = ts.get_profit_statement(self.code)
        pf.set_index(["报表日期"], inplace=True)
        cf = ts.get_cash_flow(self.code)
        cf.set_index(["报表日期"], inplace=True)
        al = ts.get_balance_sheet(self.code)
        al.set_index(["报表日期"], inplace=True)
        # 取年报的列索引，生成全期年报报表
        y_report_al = ["1231" in x for x in al.columns]
        y_report_pf = ["1231" in x for x in pf.columns]
        y_report_cf = ["1231" in x for x in cf.columns]
        self.py_y = copy.deepcopy(pf.iloc[:, y_report_pf])
        self.cf_y = copy.deepcopy(cf.iloc[:, y_report_cf])
        self.al_y = copy.deepcopy(al.iloc[:, y_report_al])
        # 求年收盘价
        y_price_index = [pd.to_datetime(x).month == 12 for x in self.original_price["date"]]
        y_price = self.original_price.loc[y_price_index, "close"]
        y_price.index = pd.to_datetime(self.original_price.loc[y_price_index, "date"])
        y_price.index = y_price.index.year
        y_price.sort_index(ascending=False, inplace=True)
        self.y_price = y_price
        fd_data_index = self.al_y.columns
        self.fd_data_index = pd.to_datetime(fd_data_index).year
        # 具体基本面数据
        self.eps = pd.to_numeric(self.py_y.loc["基本每股收益(元/股)", :])
        self.net_profit = pd.to_numeric(self.py_y.loc["五、净利润", :])
        self.withinterst_Debt = pd.to_numeric(self.al_y.loc["应付账款", :]) + pd.to_numeric(self.al_y.loc["短期借款", :]) + \
                                pd.to_numeric(self.al_y.loc["应付票据", :]) + pd.to_numeric(self.al_y.loc["其他应付款", :]) + \
                                pd.to_numeric(self.al_y.loc["一年内到期的非流动负债", :]) + pd.to_numeric(self.al_y.loc["长期借款", :]) + \
                                pd.to_numeric(self.al_y.loc["应付债券", :]) + pd.to_numeric(self.al_y.loc["长期应付款", :])
        self.lqd_Asset = pd.to_numeric(self.al_y.loc["流动资产合计", :])
        self.lqd_Debt = pd.to_numeric(self.al_y.loc["流动负债合计", :])
        self.ivtry = pd.to_numeric(self.al_y.loc["存货", :])
        self.cash = pd.to_numeric(self.al_y.loc["货币资金", :]) + pd.to_numeric(self.al_y.loc["交易性金融资产", :])
        self.total_Asset = pd.to_numeric(self.al_y.loc["资产总计", :])
        self.total_Debt = pd.to_numeric(self.al_y.loc["负债合计", :])
        self.total_Equity = self.total_Asset - self.total_Debt
        self.nwc = self.lqd_Asset - self.lqd_Debt
        self.receivables = pd.to_numeric(self.al_y.loc["应收账款", :])
        self.rev = pd.to_numeric(self.py_y.loc["营业收入", :])
        self.gross_earnings = pd.to_numeric(self.py_y.loc["营业收入", :]) - pd.to_numeric(self.py_y.loc["营业成本", :])
        self.inveset_capital = self.total_Equity + self.withinterst_Debt - self.cash
        self.stock_n = pd.to_numeric(self.al_y.loc["实收资本(或股本)", :])
        self.bps = self.total_Equity / self.stock_n
        self.sale_per_share = self.rev / self.stock_n
        # 利息与EBIT
        self.intrest = pd.to_numeric(self.py_y.loc["财务费用", :]) * 0.80
        self.EBIT = pd.to_numeric(self.py_y.loc["四、利润总额", :]) +self. intrest
        self.all_Basic_V = [self.eps,self.net_profit,self.withinterst_Debt,self.inveset_capital,
                            self.lqd_Asset,self.lqd_Debt,self.ivtry,self.cash,self.total_Asset,
                            self.total_Equity,self.total_Debt,self.nwc,self.receivables,self.rev,
                            self.gross_earnings,self.EBIT,self.intrest,self.stock_n,self.bps,self.sale_per_share]
        for x in self.all_Basic_V:
            x.index = self.fd_data_index
        # 财务比率
        # 短期偿债能力：计算流动比率、速动比率（酸性比率）、现金比率
        self.lqd_R = self.lqd_Asset / self.lqd_Debt
        self.cash_R = self.cash / self.lqd_Debt
        self.acid_test_R = (self.lqd_Asset - self.ivtry) / self.lqd_Debt

        # 长期偿债能力：计算资产负债率
        self.D_to_A = self.total_Debt / self.total_Asset
        self.B_to_E = self.total_Debt / self.total_Equity
        self.A_to_E = self.total_Asset / self.total_Equity
        self.WI_D_to_A = self.withinterst_Debt / self.total_Asset
        # 营运能力比率：计算各周转率、和营运资本
        self.inv_turnover = self.rev / self.ivtry
        self.inv_turnover_days = 365 / self.inv_turnover
        self.recvbls_turnover = self.rev / self.receivables
        self.recvbls_turnover_days = 365 / self.recvbls_turnover
        self.A_turnover = self.rev / self.total_Asset

        # 盈利性指标：ROS,ROA,ROE,ROIC,毛利率
        self.ROE = self.net_profit / self.total_Equity
        self.ROA = self.net_profit / self.total_Asset
        self.ROS = self.net_profit / self.rev
        self.ROIC = self.net_profit / self.inveset_capital
        self.PM = self.net_profit / self.rev
        self.gross_R = self.gross_earnings / self.rev

        # 计算估值指标：PE,PB,PS

        self.All_PE = self.y_price / self.eps
        self.All_PS = self.y_price / self.sale_per_share
        self.All_PB = self.y_price / self.bps

        self.all_Ratio_V = [self.lqd_R, self.cash_R, self.acid_test_R, self.D_to_A, self.A_to_E, self.B_to_E,
                            self.WI_D_to_A, self.inv_turnover, self.inv_turnover_days,self.recvbls_turnover,
                            self.recvbls_turnover_days, self.A_turnover, self.ROE, self.ROIC, self.ROS, self.ROA,
                            self.gross_R, self.PM]
        # 同行业数据
        all_idstry_class = ts.get_industry_classified()
        try:
            self.indstry_str = all_idstry_class[all_idstry_class["code"] == self.code]["c_name"].values[0]
            self.idstry_code_list = list(all_idstry_class[all_idstry_class["c_name"] == self.indstry_str]["code"].values)
        except:
            print("该股票无明显行业分类，可能是上市时间较短。")
        # 数据汇总
        All_Basic_data = pd.DataFrame(self.all_Basic_V, index=["每股收益", "净利润", "有息负债", "投入资本", "流动资产", "流动负债",
                                                               "存货", "现金", "总资产", "净资产", "总负债", "营运资本", "应收账款",
                                                               "营业收入", "毛利润", "EBIT", "利息净支出", "股本数", "每股净资产",
                                                               "每股销售收入"])
        All_Basic_data[1:-2] = All_Basic_data[1:-2] / 10000

        All_Ratio_data = pd.DataFrame(self.all_Ratio_V, index=["流动比率", "现金比率", "速动比率", "资产负债率", "权益乘数", "负债权益比",
                                                               "有息负债率", "存货周转率", "存货周转天数", "应收账款周转率", "应收账款周转天数",
                                                               "总资产周转率", "ROE", "ROIC", "ROS", "ROA", "毛利率", "销售净利率"])
        All_Ratio_data.dropna(axis=1, how="all", inplace=True)

        All_Valuation_data = pd.DataFrame([self.All_PS, self.All_PE, self.All_PB], index=["全时期PS", "全时期PE", "全时期PB"])
        All_Valuation_data.dropna(axis=1, how="all", inplace=True)
        Dupon_data = pd.DataFrame(
            [self.ROE, self.ROA, self.A_to_E, self.A_turnover, self.PM, self.total_Asset, self.total_Equity,
             self.rev, self.net_profit, ],
            index=["ROE", "ROA", "权益乘数", "总资产周转率", "销售净利率", "总资产",
                   "净资产", "营业收入", "净利润"])
        Dupon_data.dropna(axis=1, how="all", inplace=True)
        # 计算增长率
        All_Growth_data = All_Basic_data.T.pct_change(-1)
        All_Growth_data = All_Growth_data.iloc[:-1, :]
        All_Growth_data.dropna(how="all", inplace=True)

        self.All_Basic_data = All_Basic_data
        self.All_Ratio_data = All_Ratio_data
        self.All_Growth_data = All_Growth_data
        self.Dupon_data = Dupon_data
        self.All_Valuation_data = All_Valuation_data
    def Deep_Data_Median(self):
        # 1/行业利润概览
        l1 = list()
        for x in self.fd_data_index:
            z = ts.get_profit_data(x, 4)
            k = z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:].median()
            l1.append(k)
        all_idt_PF_data = pd.DataFrame(l1, index=self.fd_data_index)
        all_idt_PF_data.dropna(how="all", inplace=True)
        # 2/行业现金流量概览
        l2 = list()
        for x in self.fd_data_index:
            z = ts.get_cashflow_data(x, 4)
            k = z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:].median()
            l2.append(k)
        all_idt_CS_data = pd.DataFrame(l2, index=self.fd_data_index)
        all_idt_CS_data.dropna(how="all", inplace=True)
        # 3/行业偿债能力概览
        l3 = list()
        for x in self.fd_data_index:
            z = ts.get_debtpaying_data(x, 4)
            k = z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:].median()
            l3.append(k)
        all_idt_DP_data = pd.DataFrame(l3, index=self.fd_data_index)
        all_idt_DP_data.dropna(how="all", inplace=True)
        # 4/行业营运能力概览
        l4 = list()
        for x in self.fd_data_index:
            z = ts.get_operation_data(x, 4)
            k = z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:].median()
            l4.append(k)
        all_idt_OP_data = pd.DataFrame(l4, index=self.fd_data_index)
        all_idt_OP_data.dropna(how="all", inplace=True)
        # 5/行业成长能力概览
        l5 = list()
        for x in self.fd_data_index:
            z = ts.get_growth_data(x, 4)
            k = z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:].median()
            l5.append(k)
        all_idt_GR_data = pd.DataFrame(l5, index=self.fd_data_index)
        all_idt_GR_data.dropna(how="all", inplace=True)
        # 保存数据
        data_writer = pd.ExcelWriter("股票{}的基本面数据全览（中位数视角）.xlsx".format(self.code))

        self.All_Ratio_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面财务比率")
        self.All_Basic_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面财务数据")
        self.All_Growth_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面数据增长率")
        self.All_Valuation_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="三项估值指标")
        self.Dupon_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="杜邦分析表")

        all_idt_PF_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业利润历史")
        all_idt_CS_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业现金历史")
        all_idt_OP_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业营运历史")
        all_idt_GR_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业成长历史")
        all_idt_DP_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业偿债历史")

        data_writer.save()
    def Deep_Data_noExtrameMean(self):
        # 1/行业利润概览
        l1 = list()
        for x in self.fd_data_index:
            z = ts.get_profit_data(x, 4)
            k = noExtrameMean_DataFrame(z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:])
            l1.append(k)
        all_idt_PF_data = pd.DataFrame(l1, index=self.fd_data_index)
        all_idt_PF_data = all_idt_PF_data.iloc[:, 1:]
        all_idt_PF_data.dropna(axis =1,how="all", inplace=True)
        # 2/行业现金流量概览
        l2 = list()
        for x in self.fd_data_index:
            z = ts.get_cashflow_data(x, 4)
            k = noExtrameMean_DataFrame(z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:])
            l2.append(k)
        all_idt_CS_data = pd.DataFrame(l2, index=self.fd_data_index)
        all_idt_CS_data = all_idt_CS_data.iloc[:, 1:]
        all_idt_CS_data.dropna(axis =1,how="all", inplace=True)
        # 3/行业偿债能力概览
        l3 = list()
        for x in self.fd_data_index:
            z = ts.get_debtpaying_data(x, 4)
            k = noExtrameMean_DataFrame(z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:])
            l3.append(k)
        all_idt_DP_data = pd.DataFrame(l3, index=self.fd_data_index)
        all_idt_DP_data = all_idt_DP_data.iloc[:, 1:]
        all_idt_DP_data.dropna(axis =1,how="all", inplace=True)
        # 4/行业营运能力概览
        l4 = list()
        for x in self.fd_data_index:
            z = ts.get_operation_data(x, 4)
            k = noExtrameMean_DataFrame(z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:])
            l4.append(k)
        all_idt_OP_data = pd.DataFrame(l4, index=self.fd_data_index)
        all_idt_OP_data = all_idt_OP_data.iloc[:, 1:]
        all_idt_OP_data.dropna(axis =1,how="all", inplace=True)
        # 5/行业成长能力概览
        l5 = list()
        for x in self.fd_data_index:
            z = ts.get_growth_data(x, 4)
            k = noExtrameMean_DataFrame(z[z["code"].isin(self.idstry_code_list)].iloc[:, 1:])
            l5.append(k)
        all_idt_GR_data = pd.DataFrame(l5, index=self.fd_data_index)
        all_idt_GR_data = all_idt_GR_data.iloc[:, 1:]
        all_idt_GR_data.dropna(axis =1,how="all", inplace=True)
        # 保存数据
        data_writer = pd.ExcelWriter("股票{}的基本面数据全览(无极端平均值视角).xlsx".format(self.code))
        self.All_Ratio_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面财务比率")
        self.All_Basic_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面财务数据")
        self.All_Growth_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="基本面数据增长率")
        self.All_Valuation_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="三项估值指标")
        self.Dupon_data.T.sort_index(ascending=True).to_excel(data_writer, sheet_name="杜邦分析表")

        all_idt_PF_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业利润历史")
        all_idt_CS_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业现金历史")
        all_idt_OP_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业营运历史")
        all_idt_GR_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业成长历史")
        all_idt_DP_data.sort_index(ascending=True).to_excel(data_writer, sheet_name="行业偿债历史")

        data_writer.save()
    def Chanlun_Chart(self):
        st = ts.get_k_data(self.code)
        st["date"] = pd.to_datetime(st["date"])
        st.set_index(["date"],inplace=True)
        l1 = len(st)
        l2 = 0
        while l1 > l2:
            l1 = len(st)
            Cut_KStick(st)
            l2 = len(st)
        Shape_Top_or_Bottom(st)
        dat = extractPrice(st)
        plotConciseLine(st, dat, self.code)
        print("已完成{}的缠论走势图".format(self.code))
    def Chan_ChartandData(self):
        st = ts.get_k_data(self.code,ktype="W")
        st["date"] = pd.to_datetime(st["date"])
        st.set_index(["date"],inplace=True)
        l1 = len(st)
        l2 = 0
        while l1 > l2:
            l1 = len(st)
            Cut_KStick(st)
            l2 = len(st)
        Shape_Top_or_Bottom(st)
        dat = extractPrice(st)
        plotConciseLine(st, dat, self.code)
        print("已完成{}的缠论走势图".format(self.code))
        writer = pd.ExcelWriter("{}的缠论数据.xlsx".format(self.code))
        st.to_excel(writer, "整体数据", index=True)
        dat.to_excel(writer, "缠论笔线段数据", index=True)
        print("已保存{}的缠论数据".format(self.code))
        writer.save()









