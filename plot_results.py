import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

# def plot_error(ax, title, epsilon, support_label, mwem_label, support_error, mwem_error):
#     plt.title(title)
#
#     support_error_np = np.array([support_error[run_id] for run_id in range(total_runs)])
#     support_error_max = np.max(support_error_np, axis=0)
#     support_error_min = np.min(support_error_np, axis=0)
#     support_error_ave = np.mean(support_error_np, axis=0)
#     plt.fill_between(epsilon_list, support_error_min, support_error_max, alpha=0.1)
#     plt.plot(epsilon_list, support_error_ave, '--o', label=support_label)
#
#     for mwem_eps in mwem_epsilon_values:
#         mwem_error_np = np.array([mwem_error[run_id][mwem_eps] for run_id in range(total_runs)])
#         mwem_error_max = np.max(mwem_error_np, axis=0)
#         mwem_error_min = np.min(mwem_error_np, axis=0)
#         mwem_error_average = np.mean(mwem_error_np, axis=0)
#         plt.fill_between(epsilon_list, mwem_error_min, mwem_error_max, alpha=0.1)
#         plt.plot(epsilon_list, mwem_error_average, '-o', label=f'{mwem_label}, epsilon={mwem_eps}')
#
#
#     ax.set_xticks(epsilon_list, rotation='vertical')
#     ax.set_xlabel('epsilon budget split')
#     ax.set_ylabel('max-error')
#     ax.set_grid(linestyle='-')
#     ax.set_box(on=none)


if __name__ == '__main__':
    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('results_path', type=str, help='queries')
    args = parser.parse_args()

    results_path = args.results_path
    df = pd.read_csv(results_path)
    # print(df)

    # algos = ['fem', 'dualquery']

    algos = list(df.support_algorithm.unique())
    total_epsilon = list(df.total_epsilon.unique())

    for i,algo in enumerate(algos):
        print(f'algo = {algo}')
        df2 = df[df['support_algorithm'] == algo]
        for j,epsilon in enumerate(total_epsilon):
            plt.subplot(len(algos), len(total_epsilon), j + i*len(algos) + 1)
            plt.title('{}, eps={:.1f}'.format(algo,epsilon))
            df3 = df2[df2['total_epsilon'] == epsilon]
            epsilon_split_values = df3.support_epsilon.unique()

            G = df3.groupby('support_epsilon')
            supp_mean = G.mean()['support_error'].values
            supp_max = G.max()['support_error'].values
            supp_min = G.min()['support_error'].values
            plt.plot(epsilon_split_values, supp_mean, label=algo)
            plt.fill_between(epsilon_split_values, supp_min, supp_max, alpha=0.1)

            mwem_mean = G.mean()['mwem_error'].values
            mwem_max = G.max()['mwem_error'].values
            mwem_min = G.min()['mwem_error'].values
            plt.plot(epsilon_split_values, mwem_mean, label=algo)
            plt.fill_between(epsilon_split_values, mwem_min, mwem_max, alpha=0.1)

            if i == len(algos)-1:
                plt.xticks(epsilon_split_values, rotation='vertical')
                plt.xlabel('epsilon budget split')
            else:
                # plt.xticks(epsilon_split_values, " ")
                # plt.tick_params(axis="x", which="both", bottom=False, top=False)
                plt.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

            if j == 0:
                plt.ylabel('max-error')

            plt.grid(linestyle='-')
            plt.ylim([0,0.3])
            plt.box(on=None)
    plt.savefig('result.png')


