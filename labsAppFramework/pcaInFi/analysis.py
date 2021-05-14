# import matplotlib.pyplot as plt
# import seaborn as sns

import plotly.graph_objs as go
from plotly.offline import plot

from labsAppFramework.pcaInFi.data import *


def create_dataframe_ust():
    df = pd.read_csv("data/ust_data.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"].dt.date)
    df = df.loc[:, df.columns != '2mo']
    # df.replace("N/A", np.nan, inplace=True)
    df.set_index('date', inplace=True)
    return df


def convert_ust_tenor_to_years(ust_tenor):
    if ust_tenor[-2:] == 'mo':
        return float(ust_tenor[:-2]) / 12
    elif ust_tenor[-2:] == 'yr':
        return float(ust_tenor[:-2])


def get_ust_correlation_frame(df_ust_chg):
    # Calculate Correlation
    df_ust_corr_matrix = df_ust_chg.corr().round(2)
    for r in range(len(df_ust_corr_matrix.index)):
        for c in range(len(df_ust_corr_matrix.columns)):
            if c >= r:
                df_ust_corr_matrix.iloc[r, c] = np.nan

    return df_ust_corr_matrix


def test_normality_by_year(df_yield_chg, bucketed_years=1):
    df_yield_chg = df_yield_chg.dropna()

    years = []
    tenors = []
    p_values = []
    for year in df_yield_chg.index.year.unique():
        years_list = [year - i for i in range(bucketed_years)]
        df_yield_chg_annual = df_yield_chg.loc[df_yield_chg.index.year.isin(years_list)]
        for tenor in df_yield_chg_annual.columns:
            stat, p = normaltest(df_yield_chg_annual[tenor].values)
            years.append(year)
            tenors.append(tenor)
            p_values.append(round(p, 5))

    df_gaussian_test = pd.pivot_table(pd.DataFrame(dict(year=years, tenor=tenors, p_value=p_values)),
                                      index='tenor', columns='year', values='p_value', aggfunc='mean')
    df_gaussian_test = df_gaussian_test.loc[df_yield_chg_annual.columns, :]

    return df_gaussian_test


class PcaAnalysis():
    def __init__(self, data, normalized=False, tenors=None):
        self.data = data.dropna() if tenors is None else data[tenors].dropna()
        self.normalized = normalized
        self.tenors = tenors

        X = self.data
        n_comp = len(X.columns)
        if normalized:
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
            self.scaler = scaler

        pca = PCA(n_components=n_comp)
        pca.fit(X)
        eigen_vectors = pd.DataFrame(pca.components_).rename(columns={i: col for i, col in enumerate(self.data.columns)})

        if normalized:
            eigen_vectors_bps = pd.DataFrame(scaler.inverse_transform(eigen_vectors))
            self.eigen_vectors_bps = eigen_vectors_bps

        # variance percent of each PC
        explained_variance = pd.DataFrame(data=pca.explained_variance_ratio_) * 100
        explained_variance = explained_variance.reset_index() \
            .rename(columns={'index': 'eigen_vector', 0: 'explained_variance'}).set_index('eigen_vector')

        #transformed data
        df_transformed = pd.DataFrame(pca.transform(X))
        self.data_transformed = df_transformed

        self.pca_fitted = pca
        self.eigen_vectors = eigen_vectors
        self.explained_variance = explained_variance

    def get_eigen_plot_data(self, x, y, z, eigen_value=0):
        eigen, data = self.eigen_vectors, self.data
        eigen_x = [i for i in range(round(data[x].min()), round(data[x].max()))]
        eigen_y = [(X / eigen.loc[eigen_value, x]) * eigen.loc[eigen_value, y] for X in eigen_x]
        eigen_z = [(X / eigen.loc[eigen_value, x]) * eigen.loc[eigen_value, z] for X in eigen_x]
        df_eigen = pd.DataFrame(dict(x=eigen_x, y=eigen_y, z=eigen_z))
        df_eigen.columns = [x, y, z]
        return df_eigen


class YieldData:
    def __init__(self):
        self.data = create_dataframe_ust()
        self.data_chg = self.data.diff().iloc[1:] * 100  # move in basis points
        self.correlation_matrix = get_ust_correlation_frame(self.data_chg)
        self.gaussian_test_by_year = test_normality_by_year(self.data_chg, bucketed_years=1)
        self.gaussian_test_aggregate = \
            test_normality_by_year(self.data_chg,
                                   bucketed_years=len(self.data_chg.index.year.unique())) \
                [[self.data_chg.index.year.max()]]
        self.pca = PcaAnalysis(self.data_chg, normalized=False)
        self.pca_normalized = PcaAnalysis(self.data_chg, normalized=True)


yield_data = YieldData()
