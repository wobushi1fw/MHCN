import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import codecs
import numpy
from numpy import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas


def load_data(path):
    """
    @brief      Loads data with tag identifiers.
    @param      path  The path
    @return     List of tuples containing (tag_id, data)
    """
    data_set = list()
    with codecs.open(path) as f:
        for line in f.readlines():
            data = line.strip().split("\t")
            tag_id = int(data[0])
            flt_data = list(map(float, data[1:]))
            data_set.append((tag_id, flt_data))
    return data_set

def dist_eucl(vecA, vecB):
	"""
	@brief      the similarity function
	@param      vecA  The vector a
	@param      vecB  The vector b
	@return     the euclidean distance
	"""
	return sqrt(sum(power(vecA - vecB, 2)))

def get_closest_dist(point, centroid):
	"""
	@brief      Gets the closest distance.
	@param      point     The point
	@param      centroid  The centroid
	@return     The closest distance.
	"""
	# 计算与已有质心最近的距离
	min_dist = inf
	for j in range(len(centroid)):
		distance = dist_eucl(point, centroid[j])
		if distance < min_dist:
			min_dist = distance
	return min_dist

def kpp_cent(data_set, k):
    """
    @brief      k-means++ init centroid
    @param      data_set  The data set
    @param      k         Number of clusters
    @return     Initial centroids
    """
    data_set_array = array([entry[1] for entry in data_set])  # Convert data_set to a NumPy array
    centroid = list()
    centroid.append(data_set_array[random.randint(0, len(data_set_array))])
    d = [0 for i in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i in range(len(data_set)):
            d[i] = get_closest_dist(data_set_array[i], centroid)
            total += d[i]
        total *= random.rand()
        # Select the next centroid
        for j in range(len(d)):
            total -= d[j]
            if total > 0:
                continue
            centroid.append(data_set_array[j])
            break
    return mat(centroid)

def kpp_Means(data_set, k, dist="dist_eucl", create_cent="kpp_cent"):
    """
    @brief      k-means++ algorithm
    @param      data_set     The data set
    @param      k            Number of clusters
    @param      dist         The distance function
    @param      create_cent  The create centroid function
    @return     Centroids and cluster assignment
    """
    m = len(data_set)
    # Initialize cluster assignments for each data point
    cluster_assignment = mat(zeros((m, 2)))  # Cluster index, distance
    # Randomly initialize the initial centroids
    centroids = eval(create_cent)(data_set, k)
    cluster_changed = True
    # Iterate until convergence
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_index = -1
            min_dist = inf
            for j in range(k):
                distance = eval(dist)(data_set[i][1], centroids[j, :])
                if distance < min_dist:
                    min_dist = distance
                    min_index = j
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True
                cluster_assignment[i, :] = min_index, min_dist ** 2

        # Calculate the mean for each cluster and update centroids
        for j in range(k):
            per_data_set = [data_set[idx] for idx in nonzero(cluster_assignment[:, 0].A == j)[0]]
            centroid_data = [entry[1] for entry in per_data_set]
            centroids[j, :] = mean(centroid_data, axis=0)

    return centroids, cluster_assignment

def plot_cluster_3d(data_mat, cluste_assment, centroid):
    """
    @brief      plot 3D cluster and centroid
    @param      data_mat        The data matrix
    @param      cluste_assment  The cluste assment
    @param      centroid        The centroid
    @return
    """
    fig = plt.figure(figsize=(15, 6), dpi=80)

    # Plot original data in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data_mat[:, 0], data_mat[:, 1], data_mat[:, 2], c='blue', marker='o')
    ax1.set_title("source data", fontsize=15)

    # Plot clustered data and centroids in 3D
    ax2 = fig.add_subplot(122, projection='3d')
    k = centroid.shape[0]
    colors = [plt.cm.Spectral(each) for each in numpy.linspace(0, 1, k)]

    for i, col in zip(range(k), colors):
        per_data_set = data_mat[numpy.nonzero(cluste_assment[:, 0].A == i)[0]]
        ax2.scatter(per_data_set[:, 0], per_data_set[:, 1], per_data_set[:, 2],
                    c=[col], marker='o', edgecolors='k', s=50)

    ax2.scatter(centroid[:, 0], centroid[:, 1], centroid[:, 2],
                c='red', marker='+', s=200, label='Centroids')
    ax2.set_title("k-Means++ Cluster, k = {}".format(k), fontsize=15)
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    # # 降维
    # # 读取原始数据 ArtistEmbedding.txt
    # embedding_path = "KmeansData/ArtistEmbedding.txt"
    # data_set = numpy.loadtxt(embedding_path, delimiter='\t', skiprows=0)
    #
    # # 使用 t-SNE 进行降维
    # tsne = TSNE(n_components=3, random_state=42)
    # embedding_3d = tsne.fit_transform(data_set)
    #
    # # 将结果保存到 ArtistEmbeddingByTsne.txt
    # with open('KmeansData/ArtistEmbeddingByTsne.txt', 'w') as file:
    #     for row in embedding_3d:
    #         file.write('\t'.join(map(str, row)) + '\n')
    #
    # #合并Id与特征向量
    # #加载id文件
    # id_path="KmeansData/IdArtist.txt"
    # EmbeddingTsne_path="KmeansData/ArtistEmbeddingByTsne.txt"
    # # 读取 IdArtist.txt
    # with open(id_path, 'r') as id_file:
    #     id_data = [int(line.strip()) for line in id_file]
    # # 读取 ArtistEmbeddingByTsne.txt
    # with open(EmbeddingTsne_path, 'r') as embedding_file:
    #     embedding_data = [list(map(float, line.strip().split('\t'))) for line in embedding_file]
    # # 合并数据
    # merged_data = [[id_val] + embedding_val[:3] for id_val, embedding_val in zip(id_data, embedding_data)]
    # # 将结果保存到 ArtistEmbeddingWithId.txt
    # EmbeddingTsneId_path="KmeansData/ArtistEmbeddingByTsneWithId.txt"
    # with open(EmbeddingTsneId_path, 'w') as output_file:
    #     for row in merged_data:
    #         output_file.write('\t'.join(map(str, row)) + '\n')

    data_set = load_data("KmeansData/ArtistEmbeddingByTsneWithId.txt")
    data_mat = mat([entry[1] for entry in data_set])
    centroid, cluster_assignment = kpp_Means(data_set, 1000)
    sse = sum(cluster_assignment[:, 1])
    print("sse is ", sse)
    #OUTPUT
    with open("KmeansData/ArtistCluster.txt", "w") as output_file:
        for i in range(centroid.shape[0]):
            cluster_indices = nonzero(cluster_assignment[:, 0].A == i)[0]
            ids = [data_set[idx][0] for idx in cluster_indices]
            output_file.write(f"{ids}\n")

    print("Tag clusters have been saved to Cluster.txt.")

    plot_cluster_3d(data_mat, cluster_assignment, centroid)

    # 改ID为Value或者Name
    # 读取ArtistCluster.txt
    # with open("KmeansData/ArtistCluster.txt",'r')as ArtistCluster:
    #     artist_cluster=eval(ArtistCluster.read())







