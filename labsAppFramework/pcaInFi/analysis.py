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
        self.data = data
        self.normalized = normalized
        self.tenors = tenors

        X = data.dropna() if tenors is None else data[tenors].dropna()
        if normalized:
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
            self.scaler = scaler

        pca = PCA(n_components=5)
        pca.fit(X)
        eigen_vectors = pd.DataFrame(pca.components_).rename(columns={i: col for i, col in enumerate(data.columns)})

        if normalized:
            eigen_vectors_bps = pd.DataFrame(scaler.inverse_transform(eigen_vectors))
            self.eigen_vectors_bps = eigen_vectors_bps

        # variance percent of each PC
        explained_variance = pd.DataFrame(data=pca.explained_variance_ratio_) * 100
        explained_variance = explained_variance.reset_index() \
            .rename(columns={'index': 'eigen_vector', 0: 'explained_variance'}).set_index('eigen_vector')

        self.pca_fitted = pca
        self.eigen_vectors = eigen_vectors
        self.explained_variance = explained_variance

    def get_eigen_plot_data(self, x, y, z, eigen_value=0):
        eigen = self.eigen_vectors
        eigen.loc[0, x]
        eigen_x = [i for i in range(round(self.data[x].min()), round(self.data[x].max()))]
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

# %%


df_eigen_table = yield_data.pca.eigen_vectors
df_eigen_table = df_eigen_table.divide(df_eigen_table['30yr'], axis=0).round(2)


def get_eigen_plot_fig(yield_data, x='2yr', y='7yr', z='30yr'):
    data = yield_data.data_chg
    df_eigen_1 = yield_data.pca.get_eigen_plot_data(x, y, z, eigen_value=0)
    df_eigen_2 = yield_data.pca.get_eigen_plot_data(x, y, z, eigen_value=1)

    fig = go.Figure(
        data=[
                 go.Scatter3d(x=data[x], y=data[y], z=data[z],
                              mode='markers',
                              marker=dict(
                                  size=5,
                                  opacity=0.7),
                              name='Historic 1D Yield Changes'
                              ),
             ] + [go.Scatter3d(x=dataframe[x], y=dataframe[y], z=dataframe[z],
                               mode='lines',
                               line=dict(
                                   dash='dash',
                                   color=['red', 'blue'][eigen]),
                               name=f'Eigen Vector {eigen + 1}: {yield_data.pca.explained_variance.explained_variance[eigen]:.0f}% variance explained'
                               ) for eigen, dataframe in enumerate([df_eigen_1, df_eigen_2])],
        layout=dict(
            title='Daily UST Yield Change PCA',
            scene=dict(
                xaxis=dict(title=x),
                yaxis=dict(title=y),
                zaxis=dict(title=z)
            )
        )
    )
    return fig


fig = get_eigen_plot_fig(yield_data)
plot(fig)
