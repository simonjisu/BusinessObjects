import numpy as np
import pandas as pd
from pathlib import Path
# complexity vs score (group by tables)
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import scipy as sp

plt.style.use('ggplot')

proj_path = Path('.').resolve()
assert proj_path.name == 'BusinessObjects', f'Expected project path to be BusinessObjects, but got {proj_path.name}'

def percentile(x: pd.Series, k: float):
        return x.quantile(k)

def get_cate_gold(x: pd.Series):
    if x['need_low']:
        return 'low'
    elif x['need_mid']:
        return 'mid'
    elif x['need_high']:
        return 'high'
    
def process_df(proj_path: Path):
    df1 = pd.read_csv(proj_path / 'experiments' / 'evals' / 'spider_train_eval_plus.csv')
    df1['sample_id'] = 'train.' + df1['sample_id'].astype(str)
    df2 = pd.read_csv(proj_path / 'experiments' / 'evals' / 'spider_dev_eval_plus.csv')
    df2['sample_id'] = 'dev.' + df2['sample_id'].astype(str)
    df = pd.concat([df1, df2]).reset_index(drop=True)
    df['score_pct'] = df['score'] * 100
    df['cate_len_tbls'] = pd.cut(df['len_tbls'], bins=[0, 1, 2, 10], labels=['1', '2', '3+'])

    # filter index that which score is 0 and the complexity is high by db_id
    # current setting is set the complexity by different db_id
    df_complexity = df.groupby(['db_id'])['gold_c'].agg(['count', 'mean', lambda x: percentile(x, 0.333), lambda x: percentile(x, 0.667), 'max'])
    df_complexity.rename(columns={'<lambda_0>': '33%', '<lambda_1>': '67%'}, inplace=True)
    df['need_high'] = df.loc[:, ['db_id', 'gold_c']].apply(lambda x: x['gold_c'] > df_complexity.loc[x['db_id'], '67%'], axis=1)
    df['need_mid'] = df.loc[:, ['db_id', 'gold_c']].apply(lambda x: (x['gold_c'] <= df_complexity.loc[x['db_id'], '67%']) & (x['gold_c'] > df_complexity.loc[x['db_id'], '33%']), axis=1)
    df['need_low'] = df.loc[:, ['db_id', 'gold_c']].apply(lambda x: x['gold_c'] <= df_complexity.loc[x['db_id'], '33%'], axis=1)
    df['cate_gold_c'] = df.apply(get_cate_gold, axis=1)
    for x in ['gold', 'pred']:
        for c in ['sel', 'cond', 'agg', 'nest', 'oth']:
            c_val = df[f'{x}_c_{c}'].mean()
            bins = [-1, c_val, 1]
            labels = [f'low(0.0~{c_val:.2f})', f'high({c_val:.2f}~1.0)']
            df[f'cate_{x}_c_{c}'] = pd.cut(df[f'{x}_c_{c}'], bins=bins, labels=labels)

    return df

def num_tbls_score(df: pd.DataFrame):
    """The number of tables is not the only factor that affects the score to generate the SQL query"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.barplot(x='cate_len_tbls', y='score', data=df, ax=ax, width=0.3)
    ax.set_title('Score vs Number of Tables')
    plt.show()

def complexity_dist(df: pd.DataFrame):
    # distribution of target SQL complexity
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for i, (col, ax, n_bins) in enumerate(zip(['gold_c', 'gold_c_sel', 'gold_c_cond', 'gold_c_agg', 'gold_c_nest', 'gold_c_oth'], axes.flatten(), [10, 5, 5, 5, 5, 5])):
        min_val = df[col].min()
        max_val = df[col].max()
        val_width = max_val - min_val
        bin_width = val_width / n_bins
        sns.histplot(df[col], bins=n_bins, ax=ax, binrange=(min_val, max_val))
        ax.set_title(col)
        ax.set_xticks(np.arange(min_val-bin_width, max_val+bin_width, bin_width).round(2))
        ax.set_xticklabels(np.arange(min_val-bin_width, max_val+bin_width, bin_width).round(2), rotation=45)
    plt.tight_layout()
    plt.show()

def complexity_corr(df: pd.DataFrame):
    # correlation between complexity and score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(df.loc[:, ['score', 'len_tbls', 'gold_c', 'pred_c', 'gold_c_sel',  'gold_c_cond', 'gold_c_agg', 'gold_c_nest', 'gold_c_oth']].corr(), annot=True, ax=ax)
    ax.set_title('Correlation between Complexity and Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()

def exec_acc_two_complexities(df: pd.DataFrame):
    palette = sns.color_palette("Blues", 4).as_hex()[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    # Average Score vs Target SQL Complexity
    bb = sns.barplot(x='cate_gold_c', y='score_pct', hue='cate_len_tbls', data=df, ax=axes[0], palette=palette, width=0.5, gap=0.12, legend=True)
    bb.legend_.remove()
    axes[0].set_ylabel('Execution Accuracy(%)', fontsize=14)
    axes[0].set_xlabel('Structural Complexity($\\xi_s$)', fontsize=14)
    axes[0].set_title('Avg Acc. by different complexities')
    axes[0].tick_params(axis='both', which='major', labelsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    lgd = axes[0].legend(handles, labels, title='Num. Tables($\\xi_t$)', loc='upper right', bbox_to_anchor=(1.0, 1.0))
    lgd.get_frame().set_facecolor('white')

    # Correlation Distribution
    df_corr = df.loc[:, ['db_id', 'gold_c', 'score']].groupby('db_id')[['score', 'gold_c']].corr().unstack()['gold_c']['score']
    df_corr.rename('Pearson Correlation', inplace=True)
    sns.histplot(df_corr, bins=10, ax=axes[1], binwidth=0.10, kde=True, color=sns.color_palette("Spectral", 6).as_hex()[5])
    axes[1].set_title('Per DB Correlation($\\xi_s$ and Acc.) Distribution')
    axes[1].set_xlabel('Pearson Correlation', fontsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    title = 'Relationship between Execution Accuracy and two Complexities'
    fig.suptitle(title, fontsize=16, y=0.98)
    # fig.savefig('complexity_vs_score.pdf', bbox_inches='tight', dpi='figure', pad_inches=0.05)
    plt.show()

def exe_acc_five_features(df: pd.DataFrame):
    cols = ['cate_gold_c_sel', 'cate_gold_c_cond', 'cate_gold_c_agg', 'cate_gold_c_nest', 'cate_gold_c_oth']
    xaxis_labels = ['Selection', 'Condition', 'Aggregation', 'Nested', 'Others']
    palette = sns.color_palette("Blues", 4).as_hex()[1:]

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharey=True)
    axes.flatten()[-1].remove()
    for i, c in enumerate(cols):
        ax = axes.flatten()[i]
        sns.barplot(x=c, y='score', data=df, hue='cate_len_tbls', errorbar=('ci', 95), palette=palette, ax=ax, legend=True, dodge=True, width=0.4, gap=0.1)
        ax.set_xlabel('')
        ax.set_title(xaxis_labels[i])
        ax.set_ylabel('Execution Accuracy')

        handles, _ = ax.get_legend_handles_labels()
        vs = df[c].value_counts().values
        ls = [f'low: {vs[0]:,}', f'high: {vs[1]:,}']
        lgd = ax.legend(handles, ls, title='$\\xi_s$', loc='lower center', bbox_to_anchor=(0.51, 0.0), handlelength=0, handletextpad=0)
        lgd.get_frame().set_facecolor('white')
        # ax.legend_.remove()

    title = 'Relationship between Execution Accuracy vs Structural Complexity (by 5 Features)'
    fig.suptitle(title, fontsize=16, y=1.0)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, title='$N_{tables}$', loc='lower center', bbox_to_anchor=(0.60, 0.07))
    plt.tight_layout(pad=0.75)
    plt.show()

def two_complexity_relationship_detail(df: pd.DataFrame):
    # number of tables vs complexity (stacked bar plot)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    palette = sns.color_palette("Blues", 4).as_hex()[1:]
    df_size_len_c = df.loc[:, ['cate_len_tbls', 'cate_gold_c']].groupby(['cate_gold_c', 'cate_len_tbls'], observed=True).size().unstack()
    df_size_len_c = df_size_len_c*100 / df_size_len_c.sum(axis=0)
    df_size_len_c = df_size_len_c.reindex(['low', 'mid', 'high'])
    df_size_len_c.T.plot(kind='bar', stacked=True, ax=axes[0], color=palette)
    axes[0].set_title('Distribution of Samples')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    lgd = axes[0].legend(title='Complexity', loc='lower center', bbox_to_anchor=(0.5, -0.45), prop={'size': 8})
    lgd.get_frame().set_facecolor('white')
    axes[0].set_xlabel('Num. Tables Complexity ($\\xi_t$)')
    axes[0].set_ylabel('Percengate')

    # correlation between number of tables and complexity (scatter plot)
    palette = itemgetter(*[0, -1])(sns.color_palette("RdYlGn", 6).as_hex())
    sns.scatterplot(x='len_tbls', y='gold_c', data=df, hue='score', palette=palette, ax=axes[1])
    axes[1].set_title('Correlation by Execution Result')
    axes[1].set_ylabel('Structural Complexity ($\\xi_s$)')
    axes[1].set_xlabel('Num. Tables Complexity ($\\xi_t$)')
    axes[1].set_xticks(np.arange(0, 6))
    axes[1].set_xticklabels(np.arange(0, 6))
    lgd = axes[1].legend(title='Execution Result', loc='lower center', bbox_to_anchor=(0.5, -0.35), prop={'size': 8}, ncol=2)
    lgd.get_frame().set_facecolor('white')
    r, p = sp.stats.pearsonr(x=df['len_tbls'].values, y=df['gold_c'].values)
    m, b = np.polyfit(df['len_tbls'].values, df['gold_c'].values, 1)
    xx = np.arange(0, 6)
    axes[1].plot(xx, xx*m+b, color='black', linestyle='--')
    axes[1].text(0.78, 0.60, f'Pearson \nCorr: {r:.2f}', transform=axes[1].transAxes)

    # correlation between complexity and number of tables by each db_id
    df_corr = df.loc[:, ['db_id', 'len_tbls', 'gold_c']].groupby('db_id')[['len_tbls', 'gold_c']].corr().unstack()['gold_c']['len_tbls']
    df_corr.rename('Pearson Correlation', inplace=True)
    sns.histplot(df_corr, bins=10, ax=axes[2], binwidth=0.10, kde=True, color=sns.color_palette("Spectral", 6).as_hex()[5])
    axes[2].set_title('Per DB Correlation Distribution')
    axes[2].set_xlabel('Pearson Correlation')

    fig.suptitle('Relationship between number of two Complexities ($\\xi_s, \\xi_t$)', fontsize=16)
    plt.tight_layout()
    plt.show()

def two_complexity_relationship(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    palette = sns.color_palette("Blues", 4).as_hex()[1:]
    df_size_len_c = df.loc[:, ['cate_len_tbls', 'cate_gold_c']].groupby(['cate_len_tbls', 'cate_gold_c'], observed=True).size().unstack().loc[:, ['low', 'mid', 'high']]
    df_size_len_c = df_size_len_c*100 / df_size_len_c.sum(axis=0)
    df_size_len_c.T.plot(kind='bar', stacked=True, ax=axes[0], color=palette)
    axes[0].set_title('Distribution of Samples')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    lgd = axes[0].legend(title='Num. Tables Complexity($\\xi_t$)', loc='lower center', ncol=3, prop={'size': 8})
    lgd.get_frame().set_facecolor('white')
    axes[0].set_xlabel('Structural Complexity($\\xi_s$)', fontsize=14)
    axes[0].set_ylabel('Percengate(%)', fontsize=14)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    # correlation between complexity and number of tables by each db_id
    df_corr = df.loc[:, ['db_id', 'len_tbls', 'gold_c']].groupby('db_id')[['len_tbls', 'gold_c']].corr().unstack()['gold_c']['len_tbls']
    df_corr.rename('Pearson Correlation', inplace=True)
    sns.histplot(df_corr, bins=10, ax=axes[1], binwidth=0.10, kde=True, color=sns.color_palette("Spectral", 6).as_hex()[5])
    axes[1].set_title('Per DB Correlation ($\\xi_s$ and $\\xi_t$) Distribution')
    axes[1].set_xlabel('Pearson Correlation', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    fig.suptitle('Relationship between two Complexity ($\\xi_s, \\xi_t$)', fontsize=16)
    # fig.savefig('n_tbls_sc.pdf', bbox_inches='tight', dpi='figure', pad_inches=0.05)
    plt.tight_layout()
    plt.show()

def partial_match_score(df: pd.DataFrame, col: str='cate_len_tbls'):
    """col: cate_len_tbls, cate_gold_c"""
    title = 'Partial Match Score'

    df_len_tbls_score = df.groupby([col], observed=False)[['s_sel', 's_cond', 's_agg', 's_nest', 's_oth']].mean()
    df_draw = (df_len_tbls_score*100).fillna(0).round(6)
    df_n_qs = df.groupby([col], observed=False).size().reset_index(name='n_question').set_index('cate_len_tbls')
    cate = df_n_qs.index.tolist()
    palette = sns.color_palette("Blues", 4).as_hex()[1:]

    # Aspects and their values
    aspects = ['Selection', 'Condition', 'Aggregation', 'Nested', 'Others']
    N = len(aspects)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.rc('figure', figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, polar=True)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], aspects, color='black', size=12)
    ax.tick_params(axis='x', rotation=0)

    ax.set_rlabel_position(0)
    plt.yticks([20,40,60,80], ["20","40","60","80"], color="black", size=10)
    plt.ylim(0,100)

    for idx in range(len(df_draw)):
        values = df_draw.reset_index().loc[idx].values.tolist()[1:]
        values += values[:1]
        label = '$N_{table}$: ' + f'{cate[idx]}' + ' - ' + '$N_{samples}$: ' + f'{df_n_qs.loc[cate[idx]].values[0]:,}'
        ax.plot(angles, values, color = palette[idx], linewidth=1, linestyle='solid', label=label)
        # ax.fill(angles, values, color = palette[idx], alpha = 0.1, label=label)

    ax.set_title(title, fontsize=20, x = 0.5, y = 1.08)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24))
    plt.show()

def process_df2(df: pd.DataFrame):
    df_all = df.loc[:, ['sample_id', 'db_id', 'question', 'score', 'gold_sql', 'pred_sql', 's_sel', 's_cond', 's_agg', 's_nest', 's_oth', 'source_tables', 'len_tbls', 'cate_len_tbls', 'gold_c']].copy()

    def need_wrong(x: pd.Series, col: str):
        return (x['score'] != 1) & (x[col] == True)

    def need_correct(x: pd.Series, col: str):
        return (x['score'] == 1) & (x[col] == True)

    def need_wrong_cate_len_tbls(x: pd.Series, val: str):
        return (x['score'] != 1) & (x['cate_len_tbls'] == val)

    def need_correct_cate_len_tbls(x: pd.Series, val: str):
        return (x['score'] == 1) & (x['cate_len_tbls'] == val)

    for need_type in ['need_high', 'need_mid', 'need_low']:
        df_all[f'{need_type}|wrong'] = df_all.loc[:, ['score', need_type]].apply(need_wrong, col=need_type, axis=1)
        df_all[f'{need_type}|correct'] = df_all.loc[:, ['score', need_type]].apply(need_correct, col=need_type, axis=1)

    for cate_len_tbl in ['1', '2', '3+']:
        df_all[f'need_{cate_len_tbl}|wrong'] = df_all.loc[:, ['score', 'cate_len_tbls']].apply(need_wrong_cate_len_tbls, val=cate_len_tbl, axis=1)
        df_all[f'need_{cate_len_tbl}|correct'] = df_all.loc[:, ['score', 'cate_len_tbls']].apply(need_correct_cate_len_tbls, val=cate_len_tbl, axis=1)

    return df_all

def score_dist_num_tbls(df_all: pd.DataFrame):
    df_count = df_all.groupby(['db_id'], observed=True)[['need_1|wrong', 'need_2|wrong', 'need_3+|wrong', 'need_1|correct', 'need_2|correct', 'need_3+|correct']].sum()
    df_count.rename(columns=
        {'need_1|wrong': '1|wrong', 'need_2|wrong': '2|wrong', 'need_3+|wrong': '3+|wrong', 
        'need_1|correct': '1|correct', 'need_2|correct': '2|correct', 'need_3+|correct': '3+|correct'}, inplace=True)

    palette = itemgetter(*[2, 1, 0, -3, -2, -1])(sns.color_palette("RdYlGn", 7).as_hex())
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    df_count.plot(kind='bar', y=['1|wrong', '2|wrong', '3+|wrong', '1|correct', '2|correct', '3+|correct'], stacked=True, ax=ax, color=palette)
    ax.set_title('Score Distribution by Database and Number of Tables')
    lgd = ax.legend(loc='upper right')
    lgd.get_frame().set_facecolor('white')
    ax.set_xticks(np.arange(len(df_count.index)))
    ax.set_xticklabels(df_count.index, fontdict={'fontsize': 6})
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig('score_dist_numtbls.pdf', bbox_inches='tight', dpi='figure', pad_inches=0.05)
    plt.show()

def score_dist_structural(df_all: pd.DataFrame):
    df_count = df_all.groupby(['db_id'], observed=True)[['need_low|wrong', 'need_mid|wrong', 'need_high|wrong', 'need_low|correct', 'need_mid|correct', 'need_high|correct']].sum()
    df_count.rename(columns=
        {'need_low|wrong': 'low|wrong', 'need_mid|wrong': 'mid|wrong', 'need_high|wrong': 'high|wrong',
        'need_low|correct': 'low|correct', 'need_mid|correct': 'mid|correct', 'need_high|correct': 'high|correct'}, inplace=True)

    palette = itemgetter(*[2, 1, 0, -3, -2, -1])(sns.color_palette("RdYlGn", 7).as_hex())
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    df_count.plot(kind='bar', y=['low|wrong', 'mid|wrong', 'high|wrong', 'low|correct', 'mid|correct', 'high|correct'], stacked=True, ax=ax, color=palette)
    ax.set_title('Score Distribution by Database and Number of Tables')
    lgd = ax.legend(loc='upper right')
    lgd.get_frame().set_facecolor('white')
    ax.set_xticks(np.arange(len(df_count.index)))
    ax.set_xticklabels(df_count.index, fontdict={'fontsize': 6})
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig('score_dist_structrual.pdf', bbox_inches='tight', dpi='figure', pad_inches=0.05)
    plt.show()

def train_split_bo(proj_path: Path, df_all: pd.DataFrame):
    cols = [
        'sample_id', 'db_id', 'question', 'score', 'gold_sql', 'pred_sql', 's_sel', 's_cond', 's_agg', 's_nest', 's_oth', 'cate_len_tbls', 'cate_gold_c', 
        'need_high|wrong', 'need_high|correct', 'need_mid|wrong', 'need_mid|correct', 'need_low|wrong', 'need_low|correct', 
        'need_1|wrong', 'need_1|correct', 'need_2|wrong', 'need_2|correct', 'need_3+|wrong', 'need_3+|correct'
    ]

    df_train = df_all.loc[df_all['score'] == 1, cols]
    df_test = df_all.loc[df_all['score'] != 1, cols]
    df_train.to_csv(proj_path / 'data' / 'split_in_domain' / 'train_origin.csv', index=False)
    df_train = df_train.drop_duplicates(subset=['gold_sql'])
    df_train.to_csv(proj_path / 'data' / 'split_in_domain' / 'train.csv', index=False)
    df_test.to_csv(proj_path / 'data' / 'split_in_domain' / 'test.csv', index=False)