import matplotlib.pyplot as plt
import skimage
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from scipy.stats import ranksums
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Helper functions to handle plotting and data processing

def spatialplot(feature, featuretype, cell_positions, mask, legend=None, size=10, file=None, colourmap=None, norm=True, divnorm=None):
    assert featuretype == 'discrete' or featuretype == 'continuous'
    if featuretype == 'discrete':
        img = np.zeros_like(mask).astype(int)
        if colourmap is None:
            colourmap = plt.cm.get_cmap('tab20')

        # Create mapping for feature
        mapping = dict(zip(np.unique(feature), np.arange(0, len(np.unique(feature)))))
    elif featuretype == 'continuous':
        img = np.zeros_like(mask).astype(float)
        if colourmap is None:
            colourmap = plt.cm.get_cmap('viridis')

        # scale feature
        if norm:
            feature=np.clip(feature, np.percentile(feature, 1), np.percentile(feature, 99))
            feature=np.divide(feature - np.min(feature), np.max(feature) - np.min(feature))
        feature_max = np.max(feature)
        feature_min = np.min(feature)

    props = regionprops(mask)
    img_centroids = np.array([np.flip(regionprop.centroid) for regionprop in regionprops(mask)])

    indices = np.argmin(cdist(cell_positions, img_centroids), axis=1)
    involvedcells_labels = np.zeros(cell_positions.shape[0])

    for i in range(cell_positions.shape[0]):
        img_index = indices[i]
        img_label = props[img_index].label
        involvedcells_labels[i] = img_label
        if featuretype == 'discrete':
            img[mask==img_label] = mapping[feature[i]]
        elif featuretype == 'continuous':
            img[mask==img_label] = feature[i]

    masked_img = np.ma.masked_where(~np.isin(mask, involvedcells_labels), img)
    colourmap.set_bad(color='white')

    fig, ax = plt.subplots(1,1,figsize=np.divide(mask.shape,72), dpi=72)
    if featuretype == 'continuous':
        if divnorm is not None:
            ax.imshow(masked_img, cmap=colourmap, norm=divnorm)
        else:
            ax.imshow(masked_img, cmap=colourmap)
    else:
        img = colourmap(img)
        img[~np.isin(mask, involvedcells_labels),:] = (0.0,0.0,0.0,0.0)
        ax.imshow(img)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if featuretype == 'continuous':
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.5)
        if divnorm is None:
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=mcolors.Normalize(feature_min, feature_max), cmap=colourmap), cax=cax1)
        else:
            p_cb = plt.colorbar(mappable=cm.ScalarMappable(norm=divnorm, cmap=colourmap), cax=cax1)
        p_cb.set_label(legend, size=size)
        p_cb.ax.tick_params(labelsize=size)
    elif featuretype == 'discrete':
        colors = [ colourmap(value) for value in list(mapping.values())]
        patches = [ mpatches.Patch(color=colors[i], label=f'{list(mapping.keys())[list(mapping.values()).index(i)]}') for i in mapping.values() ]
        plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0. , prop={'size':size}, title=legend, title_fontsize=size)

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def tension_lineplot(boundary_tensions, boundary_celltypes, xlim=None):
    fig, ax = plt.subplots(1)

    # Edit boundary_tensions object for plotting
    tensions_obj = boundary_tensions.copy()
    tensions_obj['distance_to_boundary'] = tensions_obj['distance_to_boundary'].values.astype(float)

    for i, junction in tensions_obj.iterrows():
        if boundary_celltypes[1] in junction['type']:
            tensions_obj.loc[i,'distance_to_boundary'] = -junction['distance_to_boundary']

    tensions_summary = tensions_obj[tensions_obj['type']!='other'].groupby('distance_to_boundary')['tension'].agg(['mean','sem'])
    tensions_summary['ci95_hi'] = tensions_summary['mean'] + 1.96*tensions_summary['sem']
    tensions_summary['ci95_lo'] = tensions_summary['mean'] - 1.96*tensions_summary['sem']
    ax.plot(tensions_summary.index.values, tensions_summary['mean'], 'k-')
    ax.fill_between(tensions_summary.index.values, tensions_summary['ci95_lo'], tensions_summary['ci95_hi'], alpha=0.5, color='r')

    ax.vlines(0, 0, 1000)
    ax.set_ylim((0.9*np.min(tensions_summary['ci95_lo']),1.1*np.max(tensions_summary['ci95_hi'])))
    if not xlim is None:
        ax.set_xlim(xlim)
    fig.canvas.draw()
    new_labels = [label.get_text().replace('−','') for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)
    ax.set_xlabel('Distance from boundary (no. cells)')
    ax.set_ylabel('Tension')
    ax.set_title(f' {boundary_celltypes[1]}               {boundary_celltypes[0]} ')
    plt.close(fig)
    return fig

def tension_gex_lineplot(boundary_tensions, boundary_celltypes, marker_genes, gex_res, combined_data, distance_to_boundary, xlim=None, fill_type='ci95', colors=['#7fc97f','#666666']):

    # generate combined plot of tension
    # and marker gene expression as a function of distance from boundary
    # This requires a slightly different definition of 'distance'
    # for junctions - new definition is the average of the cell distances

    # Plots for gene expression

    fig, (ax, ax1) = plt.subplots(2, sharex=True)


    # Plots for tension
    # Edit boundary_tensions object for plotting
    tensions_obj = boundary_tensions.copy()
    tensions_obj['distance_to_boundary'] = tensions_obj['distance_to_boundary'].values.astype(float)


    for i, junction in tensions_obj.iterrows():
        if boundary_celltypes[1] in junction['type']:
            tensions_obj.loc[i,'distance_to_boundary'] = - junction['distance_to_boundary']
        else:
            tensions_obj.loc[i,'distance_to_boundary'] = junction['distance_to_boundary']

    tensions_summary = tensions_obj.groupby('distance_to_boundary')['tension'].agg(['mean','sem'])
    if fill_type == 'ci95':
        tensions_summary['upper_bound'] = tensions_summary['mean'] + 1.96*tensions_summary['sem']
        tensions_summary['lower_bound'] = tensions_summary['mean'] - 1.96*tensions_summary['sem']
    elif fill_type == 'sem':
        tensions_summary['upper_bound'] = tensions_summary['mean'] + tensions_summary['sem']
        tensions_summary['lower_bound'] = tensions_summary['mean'] - tensions_summary['sem']

    ax.plot(tensions_summary.index.values, tensions_summary['mean'], 'k-')
    ax.fill_between(tensions_summary.index.values, tensions_summary['lower_bound'], tensions_summary['upper_bound'], alpha=0.5, color='r')

    # extract expression data and normalise
    expression = gex_res.loc[marker_genes].values
    expression = ((expression.T-np.min(expression, axis=1))/(np.max(expression, axis=1)-np.min(expression, axis=1))).T
    expression = pd.DataFrame(expression, columns=gex_res.columns, index=marker_genes)

    # For one side of the boundary
    cells = combined_data.index.values[np.isin(combined_data['boundary_annotation'], boundary_celltypes)]
    boundary_distance_subset = distance_to_boundary.loc[cells]

    for i, cell in boundary_distance_subset.iterrows():
        if combined_data.loc[i, 'boundary_annotation'] == boundary_celltypes[1]:
            boundary_distance_subset.loc[i, 'distance'] = -cell['distance']
        else:
            boundary_distance_subset.loc[i, 'distance'] = cell['distance']

    # Plot normalised expression for both genes
    for i, gene in enumerate(marker_genes):
        gene_expression_subset = pd.DataFrame(data=np.vstack([expression.loc[gene,cells].values, boundary_distance_subset['distance'].values]).T,
                                              columns=['expression','distance_to_boundary'], index=cells)

        junctions_summary = gene_expression_subset.groupby('distance_to_boundary')['expression'].agg(['mean','sem'])

        if fill_type == 'ci95':
            junctions_summary['upper_bound'] = junctions_summary['mean'] + 1.96*junctions_summary['sem']
            junctions_summary['lower_bound'] = junctions_summary['mean'] - 1.96*junctions_summary['sem']
        elif fill_type == 'sem':
            junctions_summary['upper_bound'] = junctions_summary['mean'] + junctions_summary['sem']
            junctions_summary['lower_bound'] = junctions_summary['mean'] - junctions_summary['sem']

        ax1.plot(junctions_summary.index.values, junctions_summary['mean'], 'k-')
        ax1.fill_between(junctions_summary.index.values, junctions_summary['lower_bound'], junctions_summary['upper_bound'], alpha=0.5, color=colors[i])

    # Set overall axis labels and formatting
    ax1.set_ylim((0,0.8))
    ax1.vlines(0, 0, 1)
    ax.vlines(0, 0, 1000)
    ax.set_ylim((0.9*np.min(tensions_summary['lower_bound']),1.1*np.max(tensions_summary['upper_bound'])))
    if not xlim is None:
        ax.set_xlim(xlim)
        ax1.set_xlim(xlim)
    ax.set_xlabel('Distance from boundary (no. cells)')
    ax1.set_xlabel('Distance from boundary (no. cells)')
    ax.set_ylabel('Tension')
    ax.set_title(f' {boundary_celltypes[1]}               {boundary_celltypes[0]} ')
    ax1.set_ylabel('Normalised expression')
    ax.tick_params(bottom = False)

    fig.canvas.draw()
    new_labels = [label.get_text().replace('−','') for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)
    ax1.legend(handles=[mpatches.Patch(color=colors[0], label=marker_genes[0]),
                        mpatches.Patch(color=colors[1], label=marker_genes[1])])
    plt.close(fig)
    return fig


def gex_lineplot(gene, tensionmap_res, boundary_celltypes, distance_to_boundary, gex_res, xlim_1, xlim_2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey = ax1)

    # For one side of the boundary
    cells = tensionmap_res.index.values[np.isin(tensionmap_res['boundary_annotation'], boundary_celltypes[0])]
    boundary_distance_subset = distance_to_boundary.loc[cells]
    gene_expression_subset = pd.DataFrame(data=np.vstack([gex_res.loc[gene, cells].values, boundary_distance_subset['distance'].values]).T,
                                          columns=['expression','distance_to_boundary'], index=cells)
    junctions_summary = gene_expression_subset.groupby('distance_to_boundary')['expression'].agg(['median','sem'])
    junctions_summary['ci95_hi'] = junctions_summary['median'] + 1.96*junctions_summary['sem']
    junctions_summary['ci95_lo'] = junctions_summary['median'] - 1.96*junctions_summary['sem']

    ax1.plot(junctions_summary.index.values, junctions_summary['median'], 'k-')
    ax1.fill_between(junctions_summary.index.values, junctions_summary['ci95_lo'], junctions_summary['ci95_hi'], alpha=0.5, color='r')
    ax1.set_title(boundary_celltypes[0])
    ax1.set_xlim([np.max(junctions_summary.index.values),np.min(junctions_summary.index.values)])
    ax1.set_xlim(xlim_1)
    ax1.set_ylabel(f'{gene} expression')

    # For the other side of the boundary
    cells = tensionmap_res.index.values[np.isin(tensionmap_res['boundary_annotation'], boundary_celltypes[1])]
    boundary_distance_subset = distance_to_boundary.loc[cells]
    gene_expression_subset = pd.DataFrame(data=np.vstack([gex_res.loc[gene, cells].values, boundary_distance_subset['distance'].values]).T,
                                          columns=['expression','distance_to_boundary'], index=cells)
    junctions_summary = gene_expression_subset.groupby('distance_to_boundary')['expression'].agg(['median','sem'])
    junctions_summary['ci95_hi'] = junctions_summary['median'] + 1.96*junctions_summary['sem']
    junctions_summary['ci95_lo'] = junctions_summary['median'] - 1.96*junctions_summary['sem']

    ax2.plot(junctions_summary.index.values, junctions_summary['median'], 'k-')
    ax2.fill_between(junctions_summary.index.values, junctions_summary['ci95_lo'], junctions_summary['ci95_hi'], alpha=0.5, color='b')
    ax2.set_title(boundary_celltypes[1])
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlim([np.min(junctions_summary.index.values),np.max(junctions_summary.index.values)])
    ax2.set_xlim(xlim_2)

    # Set overall axis labels and formatting
    ax.set_xlabel('Distance from boundary',labelpad=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom = False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return

def test_interactions(lr_pairs, gex_res, cellpair_metadata, boundary_celltypes):
    res = pd.DataFrame(np.zeros([lr_pairs.shape[0], 10]), columns=['source_gene','target_gene','ab_vs_aa_pval','ab_vs_aa_stat','ab_vs_bb_pval','ab_vs_bb_stat','ba_vs_aa_pval','ba_vs_aa_stat','ba_vs_bb_pval','ba_vs_bb_stat'])
    celltype_1 = boundary_celltypes[0]
    celltype_2 = boundary_celltypes[1]
    gex_res_norm = gex_res

    for i in tqdm(range(lr_pairs.shape[0])):
        gene_1 = lr_pairs['source_genesymbol'].values[i]
        gene_2 = lr_pairs['target_genesymbol'].values[i]

        res.loc[i, 'source_gene'] = gene_1
        res.loc[i, 'target_gene'] = gene_2
        gene1_expr = pd.DataFrame(data=np.expand_dims(gex_res_norm.loc[gene_1,:].values,1), columns=['normalised'], index=gex_res_norm.columns)
        gene2_expr = pd.DataFrame(data=np.expand_dims(gex_res_norm.loc[gene_2,:].values,1), columns=['normalised'], index=gex_res_norm.columns)

        cellpair_groups = cellpair_metadata.groupby(['cell_1_celltype','cell_2_celltype'])
        # get interaction potential for a-a
        a_a_potentials = get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_1)],'cell_2'].values, gene1_expr, gene2_expr)
        b_b_potentials = get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_2)],'cell_2'].values, gene1_expr, gene2_expr)
        a_b_potentials = np.concatenate([get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_2'].values, gene1_expr, gene2_expr), \
                                         get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_2'].values, gene2_expr, gene1_expr)])
        b_a_potentials = np.concatenate([get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_2'].values, gene1_expr, gene2_expr), \
                                         get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_2'].values, gene2_expr, gene1_expr)])

        res.loc[i, 'ab_vs_aa_pval'] = ranksums(a_b_potentials, a_a_potentials).pvalue
        res.loc[i, 'ab_vs_aa_stat'] = ranksums(a_b_potentials, a_a_potentials).statistic
        res.loc[i, 'ab_vs_bb_pval'] = ranksums(a_b_potentials, b_b_potentials).pvalue
        res.loc[i, 'ab_vs_bb_stat'] = ranksums(a_b_potentials, b_b_potentials).statistic
        res.loc[i, 'ba_vs_aa_pval'] = ranksums(b_a_potentials, a_a_potentials).pvalue
        res.loc[i, 'ba_vs_aa_stat'] = ranksums(b_a_potentials, a_a_potentials).statistic
        res.loc[i, 'ba_vs_bb_pval'] = ranksums(b_a_potentials, b_b_potentials).pvalue
        res.loc[i, 'ba_vs_bb_stat'] = ranksums(b_a_potentials, b_b_potentials).statistic

    res = res.dropna()
    res['ab_likelihood'] = np.min(np.vstack([np.abs(res['ab_vs_aa_stat'].values), np.abs(res['ab_vs_bb_stat'].values)]).T, axis=1)
    res['ba_likelihood'] = np.min(np.vstack([np.abs(res['ba_vs_aa_stat'].values), np.abs(res['ba_vs_bb_stat'].values)]).T, axis=1)

    def diff_type(x, y):
        if x > 0 and y > 0:
            return 'positive'
        elif x < 0 and y < 0:
            return 'negative'
        else:
            return 'inconsistent'
    res['ab_type'] = np.vectorize(diff_type)(res['ab_vs_aa_stat'],res['ab_vs_bb_stat'].values)
    res['ba_type'] = np.vectorize(diff_type)(res['ba_vs_aa_stat'],res['ba_vs_bb_stat'].values)
    return res.dropna()

def get_potential(group1_cells, group2_cells, gene1_expr_norm, gene2_expr_norm):
    interaction_potential = gene1_expr_norm.loc[group1_cells, 'normalised'].values * gene2_expr_norm.loc[group2_cells, 'normalised'].values
    return interaction_potential

def generate_boxplot(source_gene, target_gene, gex_res, cellpair_metadata, boundary_celltypes, celltype_alias=None):
    celltype_1 = boundary_celltypes[0]
    celltype_2 = boundary_celltypes[1]
    gex_res_norm = gex_res

    gene1_expr = pd.DataFrame(data=np.expand_dims(gex_res_norm.loc[source_gene,:].values,1), columns=['normalised'], index=gex_res_norm.columns)
    gene2_expr = pd.DataFrame(data=np.expand_dims(gex_res_norm.loc[target_gene,:].values,1), columns=['normalised'], index=gex_res_norm.columns)

    cellpair_groups = cellpair_metadata.groupby(['cell_1_celltype','cell_2_celltype'])
    # get interaction potential for a-a
    a_a_potentials = get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_1)],'cell_2'].values, gene1_expr, gene2_expr)
    b_b_potentials = get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_2)],'cell_2'].values, gene1_expr, gene2_expr)
    a_b_potentials = np.concatenate([get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_2'].values, gene1_expr, gene2_expr),
                                     get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_2'].values, gene2_expr, gene1_expr)])
    b_a_potentials = np.concatenate([get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_2, celltype_1)],'cell_2'].values, gene1_expr, gene2_expr),
                                     get_potential(cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_1'].values, cellpair_metadata.loc[cellpair_groups.groups[(celltype_1, celltype_2)],'cell_2'].values, gene2_expr, gene1_expr)])
    p = sns.boxplot(data=[a_a_potentials, b_b_potentials, a_b_potentials, b_a_potentials], fliersize=0)
    sns.stripplot(data=[a_a_potentials, b_b_potentials, a_b_potentials, b_a_potentials])
    if celltype_alias is None:
        p.set_xticklabels([f'{celltype_1}-{celltype_1}',f'{celltype_2}-{celltype_2}',f'{celltype_1}-{celltype_2}',f'{celltype_2}-{celltype_1}'])
    else:
        p.set_xticklabels([f'{celltype_alias[0]}-{celltype_alias[0]}',f'{celltype_alias[1]}-{celltype_alias[1]}',f'{celltype_alias[0]}-{celltype_alias[1]}',f'{celltype_alias[1]}-{celltype_alias[0]}'])

    p.set(ylabel='Junction interaction potential', title=f'{source_gene} -> {target_gene}')
    return p