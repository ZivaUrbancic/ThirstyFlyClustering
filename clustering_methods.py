# Libraries:
import csv
import kcluster
import numpy as np
import seaborn as sns
import networkx as nx
import umap
import umap.plot
import gudhi as gd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy.stats import percentileofscore
from scipy.spatial.distance import pdist, squareform
from pyballmapper import BallMapper


cmap=plt.get_cmap('hsv');
plt.set_cmap(cmap);


def load_data(data_file, peptidergic=False):
    """
    A method for reading data from csv file. Returns data in the form of 2D array M,
    and two lists of strings, neurons and genes, with titles of rows and columns
    respectively.

    Input:
       data_file: string name of a csv file with the dataset

    Output:
        M: 2D array of data where rows correspond to different
           neurons and columns correspond to different genes.
        neurons: list of strings, names of neurons (rows).
        genes: list of strings, the names of genes (columns).
    """

    with open(data_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        counter = 0
        genes = []
        M = []
        for row in spamreader:
            if counter == 0:
                neurons = row[0].replace('"', '')
                if peptidergic:
                    neurons = neurons.split(',')
                else:
                    neurons = neurons.split(',')[1:]
                #print(neurons)
                counter += 1
            else:
                gene = row[0].split(',')
                label = gene[0].replace('"','')
                genes.append(label)
                entry = [int(g) for g in gene[1:]]
                M.append(entry)
    M = np.array(M).T
    return M, neurons, genes


def trim_data(M, neurons, genes, threshold_expression = 0.005):
    n_neurons = len(neurons)
    n_genes = len(genes)
    assert (len(M) == n_neurons) and (len(M[0,:])==n_genes)
    # Remove genes that are almost never expressed:
    keep_cols = [i for i, count in enumerate(np.count_nonzero(M, axis=0)) if count > threshold_expression*n_neurons]
    # Remove rows with tiny expression count:
    keep_rows = [i for i, count in enumerate(np.count_nonzero(M, axis=1)) if count > threshold_expression*n_genes]
    M_updated = M[:, keep_cols]
    n_updated = np.array(neurons)[keep_rows]
    g_updated = np.array(genes)[keep_cols]
    return M_updated[keep_rows,:], n_updated, g_updated, keep_cols, keep_rows


def get_representatives(matrix, thresholds, exclusive=False):
    """
    Returns a dictionary with keys 'Dopamine', 'Serotonine', 'Octopamine' and
    'Tyramine', each with a list of ints (ids of points that are representatives 
    of the class) as an entry.
    This method will be updated to take a general matrix.
    
    Input:
        matrix: Nxlen(thresholds) numpy array, where the columns appear in order:
                ('Fer2',) 'Trh', 'ple', 'Tdc2', 'DAT', 'SerT', 'Tbh' (can be used with
                iM type of matrices). If fer2 is False, column of 'Fer2' is not given.
        thresholds: list of ints, threshold for the significance of expression
                    of chosen genes.

    Output:
        ids: dictionary with keys 'Dopamine', 'Serotonine', 'Octopamine' and
             'Tyramine', each with a list of ints.
        bad_neurons: list of int ids of neurons that appear in more than one class.
    """

    #classes = [[] for i in range(len(matrix))]
    truths = np.array([matrix[i, :]>thresholds for i in range(len(matrix))])
    expression = np.array([matrix[i, :]>np.zeros(shape=thresholds.shape) for i in range(len(matrix))])
    ids = {'Dopamine':[], 'Serotonin':[], 'Octopamine':[], 'Tyramine':[]}
    #no_class=0
    bad_neurons=[]
    for i in range(len(matrix)):
        row = truths[i, :]
        e = expression[i, :]
        st=0
        tags = []
        # Dopamine:
        #if (row[-5] and row[-3]):
        #if (exclusive and row[-5] and row[-3] and not e[-1] and not e[-2] and not e[-4] and not e[-6]) or (not exclusive and row[-5] and row[-3]):
        if row[-5] and row[-3] and not e[-1] and not e[-2] and not e[-4] and not e[-6]:
            #ids['Dopamine']+=[i]
            tags += ["Dopamine"]
            st+=1
        # Serotonin:
        #if (row[-2] and row[-6]):
        #if (exclusive and row[-2] and row[-6] and not e[-1] and not e[-3] and not e[-4] and not e[-5]) or (not exclusive and row[-2] and row[-6]):
        if row[-2] and row[-6] and not e[-1] and not e[-3] and not e[-4] and not e[-5]:
            #ids['Serotonine']+=[i]
            tags += ["Serotonin"]
            st+=1
        # Octopamine:
        #if (row[-4] and row[-1]):
        #if (exclusive and row[-1] and row[-4] and not e[-5] and not e[-2] and not e[-3] and not e[-6]) or (not exclusive and row[-4] and row[-1]):
        if row[-1] and row[-4] and not e[-5] and not e[-2] and not e[-3] and not e[-6]:
            #ids['Octopamine']+=[i]
            tags += ["Octopamine"]
            st+=1
        # Tyramine:
        #if (row[-4] and not row[-1]):
        #if (exclusive and row[-4] and not e[-3] and not e[-1] and not e[-2] and not e[-5] and not e[-6]) or (not exclusive and row[-4] and not row[-1]):
        if row[-4] and not e[-3] and not e[-1] and not e[-2] and not e[-5] and not e[-6]:
            #ids['Tyramine']+=[i]
            tags += ["Tyramine"]
            st+=1
        if st>1:
            bad_neurons+=[i]
        elif st==1:
            ids[tags[0]] += [i]
    return ids, bad_neurons


##############################################################################################
# Inter/Intra label comparisson
##############################################################################################

def split_data_labelwise(labels, matrika, nr_labels=4):
    """
    Groups datapoints based on cluster membership.

    Input:
        labels: list of ints, for each point includes its cluster's id.
        matrika: 2D array of datapoints to be split, each row is a point.
        nr_labels: int, number of different cluster ids (default 4).

    Output: splits, max_value
        splits: dict with cluster ids as keys and 2D array of points in
                respective clusters as values.
    """

    splits = {}
    for label in range(nr_labels):
        ids = [i for i in range(len(labels)) if labels[i] == label]
        splits[label] = matrika[ids, :]
    return splits


def visualize_umap(mapper, labels, annotate={}, legend_text={}, title=None, loc_clusterlegend=1, loc_replegend=5, cluster_labels=None, axis=None):
    """
    Method for visualisation of umap plots.

    Input:
        mapper: umap.umap_ object, fit of matrix to plot.
        labels: list of ints, labels of the dataset.
        annotate: dict with keys being the matplotlib marker names and values being indices
                  of points in 'matrix' to mark with said marker (default {}).
        legend_text: dict with keys being the matplotlib marker names and values being the
                     string descriptors of what markers are marking, to be used in plot legend
                     (default {}).
        title: string or None, plot title to be displayed (default None).
        loc_clusterlegend: matplotlib.pyplot location string or location code for positioning of
                           legend with cluster colors (default 1 ('upper right')).
        loc_replegend: matplotlib.pyplot location string or location code for positioning of
                       legend with annotation markers (default 1 ('upper right')).
        cluster_labels: list of strings, labels for each cluster (default None).

    Output: (plot)
    """ 
    
    n_clusters = max(labels)+1
    norm = pltc.Normalize(vmin=0, vmax=n_clusters)
    cs = cmap(norm([i for i in range(n_clusters)]))
    if axis is None:
        _, ax = plt.subplots(figsize=(8,8), dpi=80)
    else:
        ax = axis
    # xs = mapper.embedding_[:,0]
    # ys = mapper.embedding_[:,1]
    # if cluster_labels is not None:
    #     print("Required cluster labels are: ", cluster_labels)
    #     for i in range(n_clusters):
    #         members = [j for j in range(len(xs)) if labels[j]==i]
    #         ax.scatter([xs[j] for j in members], [ys[j] for j in members], c=cs[i], s=5, label=cluster_labels[i])
    #         print("Cluster ", i, " has ", len(members), " members and label ", cluster_labels[i])
    # else:
    ax.scatter(mapper.embedding_[:,0], mapper.embedding_[:,1], c=[cs[cid] for cid in labels], s=5)

    legend2_elts = []
    assert(len(annotate.keys())==len(legend_text.keys()))
    for key in annotate.keys():
        llabels = np.array([labels[i] for i in annotate[key]]) 
        localM = mapper.embedding_[annotate[key],:]
        color_key=[cs[l] for l in llabels]
        ax.scatter(localM[:,0], localM[:,1], c=np.array(color_key), marker=key, s=50, linewidth=1, edgecolor="black")
        legend_elt = mlines.Line2D([], [], color='black', marker=key, linestyle='None',
                          markersize=8, linewidth=1, label=legend_text[key])
        legend2_elts += [legend_elt]
    
    if cluster_labels is not None:
        legend1_elts = [Patch(facecolor=cs[i], label=cluster_labels[i]) for i in range(n_clusters)]
    else:
        legend1_elts = [Patch(facecolor=cs[i], label=str(i)) for i in range(n_clusters)]
    legend1 = ax.legend(handles=legend1_elts, loc=loc_clusterlegend, fontsize=13)
    ax.add_artist(legend1)
    if len(legend2_elts)>0:
        legend2 = ax.legend(handles=legend2_elts, loc=loc_replegend)
        ax.add_artist(legend2)
    if title is not None:
        ax.set_title(title, fontsize=15)
    
    ax.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False) 
    if axis is None:
        plt.show()
    else:
        return ax


def compute_and_visualize_umap(matrix, labels, random_state=24000, metric='euclidean', annotate={}, legend_text={}, title=None, loc_clusterlegend=1, loc_replegend=5, cluster_labels=None, axis=None):
    """
    Method for visualisation of umap plots.

    Input:
        matrix: 2D numpy array of data.
        labels: list of ints, labels of the dataset.
        random_state: int, random state to use in UMAP (defaut 24000).
        annotate: dict with keys being the matplotlib marker names and values being indices
                  of points in 'matrix' to mark with said marker (default {}).
        legend_text: dict with keys being the matplotlib marker names and values being the
                     string descriptors of what markers are marking, to be used in plot legend
                     (default {}).
        title: string or None, plot title to be displayed (default None).
        loc_clusterlegend: matplotlib.pyplot location string or location code for positioning of
                           legend with cluster colors (default 1 ('upper right')).
        loc_replegend: matplotlib.pyplot location string or location code for positioning of
                       legend with annotation markers (default 1 ('upper right')).
        cluster_labels: list of strings, labels for each cluster (default None).

    Output: (plot)
        mapper: umap.umap_ object, fit of matrix.
    """ 

    mapper = umap.UMAP(random_state=random_state, metric=metric).fit(matrix)
    if axis == None:
        visualize_umap(mapper, labels,
                    annotate=annotate,
                    title=title,
                    legend_text=legend_text,
                    loc_clusterlegend=loc_clusterlegend,
                    loc_replegend=loc_replegend,
                    cluster_labels=cluster_labels,
                    axis=axis)
        return mapper
    else:
        ax = visualize_umap(mapper, labels,
                    annotate=annotate,
                    title=title,
                    legend_text=legend_text,
                    loc_clusterlegend=loc_clusterlegend,
                    loc_replegend=loc_replegend,
                    cluster_labels=cluster_labels,
                    axis=axis)
        return mapper, ax


def compute_kcluster(matrix, n_clusters, n_neighbors=4, metric='euclidean', distance_matrix=None):
    """
    Method for computation of kcluster results.

    Input:
        matrix: 2D numpy array of data.
        n_clusters: int, requested number of clusters.
        n_neighbors: lower limit of members for birth of a cluster (default 4).
        metric: 'euclidean', 'cosine' or 'corrcoef', chioce of metric (default 'euclidean').
        distance_matrix: 2D numpy array, precomputed distance matrix (default None).

    Output:
        cluster_ids: list, for each row in the matrix, int id of the cluster it is a
                     member of.
        PD: numpy.ndarray, the persistence diagram computed with kcluster.
        D: NxN distance matrix where N is the number of rows in matrix.
    """ 

    if distance_matrix is not None:
        D = distance_matrix
    else:
        if metric =='corrcoef':
            N = len(matrix)
            D = np.ones((N,N))-np.corrcoef(matrix)
        else:
            D = squareform(pdist(matrix, metric=metric))
        print("Constructed the distance matrix.")
    PD,F,E = kcluster.persistenceDiagram(D,k=n_neighbors,return_filtration=True)
    print("Constructed kcluster PD.")
    try:
        alpha = kcluster.getThreshold(PD,n_clusters)
    except Exception as e:
        txt = str(e).split()[-1]
        n_clusters = int(txt)-1
        alpha = kcluster.getThreshold(PD, n_clusters)
    clstrs_multiplicative, clstr_list = kcluster.getClusters(F,E,alpha)
    cluster_ids = [clstr_list[i] for i in clstrs_multiplicative]
    print("Clustered.")
    return cluster_ids, PD, D


def visualize_kcluster(matrix, n_clusters, n_neighbors=4, random_state=2400, annotate={}, legend_text={}, metric='euclidean', title=None, loc_clusterlegend=1, loc_replegend=5, distance_matrix=None, mapper=None, cluster_labels=None, axis=None):
    """
    Method for computation and visualisation of kcluster results using umap.

    Input:
        matrix: 2D numpy array of data.
        n_clusters: int, requested number of clusters.
        n_neighbors: lower limit of members for birth of a cluster (default 4).
        random_state: int, random state (defaut 2400).
        annotate: dict with keys being the matplotlib marker names and values being indices
                  of points in 'matrix' to mark with said marker (default {}).
        legend_text: dict with keys being the matplotlib marker names and values being the
                     string descriptors of what markers are marking, to be used in plot legend
                     (default {}).
        metric: 'euclidean' or 'corrcoef', chioce of metric (default 'euclidean').
        title: string or None, plot title to be displayed (default None).
        loc_clusterlegend: matplotlib.pyplot location string or location code for positioning of
                            legend with cluster colors (default: 1 ('upper right')).
        loc_replegend: matplotlib.pyplot location string or location code for positioning of
                          legend with annotation markers (default: 5 ('right')).
        distance_matrix: 2D numpy array, precomputed distance matrix (default None).
        mapper: umap.umap_ object, fit of matrix (default None).
        cluster_labels: list of strings, labels for each cluster (default None).

    Output: (plot)
        cluster_ids: list, for each row in the matrix, int id of the cluster it is a
                     member of.
        mapper: umap.umap_ object, fit of matrix.
        PD: numpy.ndarray, the persistence diagram computed with kcluster.
        D: NxN distance matrix where N is the number of rows in matrix.
    """ 

    cluster_ids, PD, D = compute_kcluster(matrix, n_clusters, n_neighbors, metric, distance_matrix)
    
    if mapper is None and axis is None:
        mapper = compute_and_visualize_umap(matrix, cluster_ids,
                                            random_state=random_state,
                                            metric=metric,
                                            annotate=annotate,
                                            title=title,
                                            legend_text=legend_text,
                                            loc_clusterlegend=loc_clusterlegend,
                                            loc_replegend=loc_replegend,
                                            cluster_labels=cluster_labels)
    elif mapper is None:
        mapper,ax = compute_and_visualize_umap(matrix, cluster_ids,
                                            random_state=random_state,
                                            metric=metric,
                                            annotate=annotate,
                                            title=title,
                                            legend_text=legend_text,
                                            loc_clusterlegend=loc_clusterlegend,
                                            loc_replegend=loc_replegend,
                                            cluster_labels=cluster_labels,
                                            axis=axis)
    elif axis is None:
        visualize_umap(mapper, cluster_ids,
                       annotate=annotate,
                       title=title,
                       legend_text=legend_text,
                       loc_clusterlegend=loc_clusterlegend,
                       loc_replegend=loc_replegend,
                       cluster_labels=cluster_labels)
    else:
        ax = visualize_umap(mapper, cluster_ids,
                       annotate=annotate,
                       title=title,
                       legend_text=legend_text,
                       loc_clusterlegend=loc_clusterlegend,
                       loc_replegend=loc_replegend,
                       cluster_labels=cluster_labels,
                       axis=axis)
    if axis is None:
        return cluster_ids, mapper, PD, D
    else:
        return cluster_ids, mapper, PD, D, ax


def PD_plot(matrix, title=None):
    """
    Method for computation and visualisation of persistence diagram, given an
    adjacency matrix.

    Input:
        matrix: 2D numpy array of data.
        title: string, title for the plot (default "Persistence Diagram").
        
    Output: (plot)
    """ 

    D = squareform(pdist(matrix))
    VR = gd.RipsComplex(distance_matrix=D)
    simplex_tree = VR.create_simplex_tree(max_dimension=2)
    B = simplex_tree.persistence()
    ax = gd.plot_persistence_diagram(B);
    if title is not None:
        ax.set_title(title);


def clusterwise_expression(matrix, cluster_dict, gene_ids, method='median'):
    """
    For each gene whose index is in 'gene_ids' evaluate the strength of expression
    on each cluster in 'cluster_dict'. To evaluate the strength we compute a threshold
    (either median if 'method="median"' or mean if 'mehod="mean"') for the expression
    of the gene over all clusters and compute the percentage of neurons in a cluster
    that are expressed higher than this threshold for each cluster individually.

    Input:
        matrix: 2D numpy array of data.
        cluster_dict: dictionary with cluster labels as keys and a list of indices of
                      points in the cluster.
        gene_ids: list of ints, indices of genes we wish to compute clusterwise
                  expression for.
        method: 'median', 'mean' or list. A method of obtaining thresholds. If we want
                custom thresholds, we pass a list of the same length as gene_ids.
                (Default 'median').

    Output:
        results: a dict with keys the same as cluster_dict and values the computed
                 percentages.
    """

    results={}
    for key in range(max(cluster_dict.keys())+1):
        results[key] = np.zeros(np.array(gene_ids).shape)
        #cluster = cluster_dict[key] ###### Added this because later changes
    for j, gene in enumerate(gene_ids):
        genevec=matrix[:,gene]   ##### Changed this from M[:, gene]. Might be problematic later on
        if method == 'median':
            overthresholds=[int(i) for i in genevec>np.median(genevec)]
        elif method == 'mean':
            overthresholds=[int(i) for i in genevec>np.mean(genevec)]
        elif type(method)==list:
            if j==0:
                print("method type is list, this is a custom threshold")
            assert len(method)==len(gene_ids)
            overthresholds=[int(i) for i in genevec>method[j]]
        else:
            print("Method ", method, " is not supported by clusterwise_expression.")
            break
        for key in cluster_dict.keys():
            cluster = cluster_dict[key]
            cluster_mean = np.mean([overthresholds[i] for i in cluster]) #This is the percentage of neurons in cluster for which the expression is higher than median.
            results[key][j] = cluster_mean
            #print(results[key])
            #print("    ", key, ": ", cluster_mean)
    return results



def convert_cluster_results_from_dict_to_list(cluster_dict, points_list):
    """
    Converts a dictionary of clustering results (labels as keys, list of neuron indices
    as values) into a list with, for each point, the number of cluster it belongs to.

    Input:
        cluster_dict: a dictionary of clustering results (labels as keys, list of neuron
                      indices as values).
        points_list: list of neuron indices for all neurons in different clusters.

    Output:
        cluster_list: a list with, for each point, the number of cluster it belongs to.
    """
    labels1 = list(cluster_dict.keys())

    cluster_list = np.zeros(len(points_list))
    for i, entry in enumerate(points_list):
        for j in range(len(labels1)):
            if entry in cluster_dict[labels1[j]]:
                cluster_list[i] = j
                break
    return cluster_list

def convert_list_to_dict(cluster_list):
    ids = set(cluster_list)
    cluster_dict = {}
    for i in ids:
        cluster_dict[i] = [j for j, c in enumerate(cluster_list) if c==i]
    return cluster_dict


def table_clusterwise_expression(matrix, cluster_dict, gene_ids, row_labels, method='median'):
    """
    Returns a stylized dataframe for visualization of clusterwise expression.

    Input:
        matrix: 2D numpy array of data.
        cluster_dict: dictionary with cluster labels as keys and a list of indices of
                      points in the cluster.
        gene_ids: list of ints, indices of genes we wish to compute clusterwise
                  expression for.
        row_labels: list of strings, names of genes with indices in gene_ids.
        method: 'median', 'mean' or list. A method of obtaining thresholds. If we want
                custom thresholds, we pass a list of the same length as gene_ids.
                (Default 'median').

    Output:
        s: stylized pandas dataframe.
    """

    results = clusterwise_expression(matrix, cluster_dict=cluster_dict, gene_ids=gene_ids, method=method)
    df = pd.DataFrame(results, index=row_labels)
    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient(cmap=cm)
    return s, df


def percentilize(matrix):
    """
    Take each column in the matrix and map each of its entries to the corresponding percentile of
    the column.
    
    Input:
        matrix: 2D np.array to percentilize.
        
    Output:
        2D np.array of shape matrix.shape.
    """

    percentilized = np.zeros(matrix.shape)
    for i in range(len(matrix[0,:])):
        column = matrix[:,i]
        percentilized[:,i] = percentileofscore(column, column, kind='strict')
    return percentilized


def validation_on_representatives(clustering, representatives_dict):
    results = {}
    for key in representatives_dict.keys():
        clusters = [clustering[i] for i in representatives_dict[key]]
        results[key] = [len([j for j, c in enumerate(clusters) if c==i]) for i in range(max(clustering)+1)]
    df = pd.DataFrame(results, index=[i for i in range(max(clustering)+1)])
    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient(cmap=cm)
    return s, df

def ballmapper_plot(bm_object, n_clusters, colorkey, title=None, loc=1, remove_outliers=False):
    norm = pltc.Normalize(vmin=0, vmax=n_clusters)
    cs = cmap(norm([i for i in range(n_clusters)]))
    legend_elements = [Patch(facecolor=cs[i], label=str(i)) for i in range(n_clusters)]
    graph=bm_object.Graph.copy()

    if remove_outliers:
        in_edges = set()
        for edge in graph.edges:
            in_edges = in_edges.union(set(edge))
        print("Removed", len(graph.nodes) - len(in_edges), "outliers.")
        graph.remove_nodes_from([g for g in graph.nodes if g not in in_edges])

    plt.figure(figsize=(8,8), dpi=80)
    node_size_l = [2*len(bm_object.points_covered_by_landmarks[idx]) for idx in range(len(graph.nodes))]
    #node_size_l = [5 for idx in range(len(graph.nodes))]
    node_color_l = [cs[colorkey[node]] for node in graph.nodes]
    
    nx.draw_networkx(graph,
                     pos=nx.spring_layout(bm_object.Graph, seed=24000),
                     with_labels=False,
                     node_size=node_size_l,
                     node_color=node_color_l)

    plt.legend(handles=legend_elements, loc=loc)
    if title is not None:
        plt.title(title, fontsize=18, y=1.02);


def get_plot_annotations(cluster_dict, markers=['o', '>', 'D', 's', 'P', '*', 'X', 'v', '1'], legend_suffix=""):
    assert len(cluster_dict.keys()) <= len(markers)
    annotation_dict = {}
    legend_text = {}
    for i, key in enumerate(cluster_dict.keys()):
        annotation_dict[markers[i]] = cluster_dict[key]
        if legend_suffix=="":
            legend_text[markers[i]] = "Cluster "+str(key)
        else:
            legend_text[markers[i]] = "Cluster "+str(key)+legend_suffix
    return annotation_dict, legend_text


def latexify_table(df, times_100=True, color_pallete="seagreen", threshold_max=0.9, threshold_min=0.1, format_nrs="{:,.1f}", column_width='50px', save_to=None):
    if times_100:
        df = df*100
        threshold_max = threshold_max*100
        threshold_min = threshold_min*100
    cm = sns.light_palette(color_pallete, as_cmap=True)
    dfproj = df[df.max(axis=1)>=threshold_max]
    dfproj = dfproj[dfproj.min(axis=1)<=threshold_min]
    s = dfproj.style.background_gradient(cmap=cm, axis=None)
    columns = list(df.columns)
    s.format({key: format_nrs.format for key in columns})
    s.set_properties(subset=columns, **{'width': column_width})
    s.set_properties(**{'text-align': 'center'})
    latex_table = s.to_latex(
        convert_css=True,
        hrules=True,
        caption="Placeholder caption",
        label="tab:new_table",
        column_format = 'c'*(len(columns)+1)
        )
    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(latex_table)
    return latex_table, s