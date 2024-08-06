# Mateus Mesquita de Pádua 11921ECP011
import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Função para atribuir cada ponto ao cluster mais próximo
def assign_points(data, centroids):
    clusters = []
    for point in data:
        distances = [distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters

# Função para atualizar a posição dos centroides
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(np.zeros(data.shape[1]))
    return new_centroids

def calculate_sse(data, clusters, centroids):
    sse = 0
    for i, point in enumerate(data):
        centroid = centroids[clusters[i]]
        sse += np.sum((point - centroid) ** 2)
    return sse

# Dados fornecidos
data = np.array([
    [0.0402, -0.0050],
    [2.3253, 3.2499],
    [2.0928, 3.1758],
    [1.4135, 3.0197],
    [3.2312, 1.7430],
    [2.3027, 0.9334],
    [2.6697, 1.8404],
    [1.7302, -0.6967],
    [2.2823, 2.7132],
    [2.5000, 1.5786],
    [7.7860, 7.7290],
    [8.2139, 8.1269],
    [6.6568, 8.0361],
    [8.3078, 8.3540],
    [6.9500, 7.9605],
    [9.4215, 8.0450],
    [8.5817, 9.3543],
    [8.5182, 7.5897],
    [8.7232, 10.7074],
    [8.9482, 9.7707],
    [0.1639, 8.6602],
    [0.3333, 8.8842],
    [0.2196, 8.3855],
    [2.7472, 8.4164],
    [1.9408, 8.3111],
    [2.3784, 10.0452],
    [0.4083, 8.6337],
    [1.6094, 7.1453],
    [1.7449, 10.1956],
    [1.8532, 8.7684],
    [9.3763, 1.6914],
    [8.5099, 3.1385],
    [7.9898, 2.0037],
    [9.0933, 2.8751],
    [7.8167, 0.7880],
    [10.9693, 4.1131],
    [8.8949, 2.6439],
    [9.6476, 2.6863],
    [8.8764, 3.2085],
    [9.0019, -0.1913]
])

k_values = range(1, 11)  # Testando k de 1 a 10
sse_values = []

for k in k_values:
    # Inicialização dos centroides
    np.random.seed(42)
    centroids = np.random.randn(k, data.shape[1])
    
    # Execução do algoritmo K-means
    max_iterations = 100
    for _ in range(max_iterations):
        old_centroids = centroids.copy()
        
        clusters = assign_points(data, centroids)
        
        centroids = update_centroids(data, clusters, k)
        
        if np.allclose(old_centroids, centroids):
            break
    
    # Cálculo do SSE
    sse = calculate_sse(data, clusters, centroids)
    sse_values.append(sse)

plt.plot(k_values, sse_values, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Erro Quadrático Total (SSE)')
plt.title('Curva do Erro Quadrático Total para os dados')
plt.show()
