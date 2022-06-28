from simbsig.neighbors import NearestNeighbors as NearestNeighborsOwn
from simbsig.neighbors import KNeighborsClassifier as KNeighborsClassifierOwn
from simbsig.neighbors import KNeighborsRegressor as KNeighborsRegressorOwn
from simbsig.neighbors import RadiusNeighborsClassifier as RadiusNeighborsClassifierOwn
from simbsig.neighbors import RadiusNeighborsRegressor as RadiusNeighborsRegressorOwn
from simbsig.cluster import MiniBatchKMeans as MiniBatchKMeansOwn
from simbsig.decomposition import PCA as PCAOwn
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import numpy as np

# import matplotlib.pyplot as plt

if __name__ == '__main__':

    KNN = NearestNeighbors()

    #a = np.random.rand(15, 5)
    #b = np.random.rand(10, 5)

    samples = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5], [1, 1, 1]])
    queries = np.array([[1., 1., 1.], [0., .5, 0]])

    samples = np.random.rand(15, 5)
    queries = np.random.rand(10, 5)


    ###################################################################################################################
    #### Compare NearestNeighbors
    print("-------------------------------------------------------------",
          "\n--------------- compare NearestNeighbors --------------------",
          "\n-------------------------------------------------------------")

    #### sklearn's kneighbors
    print('\n--------sklearns kneighbors--------------')
    neigh = NearestNeighbors(n_neighbors=4,algorithm='brute')
    neigh.fit(samples)
    dist_sk, neighb_sk = neigh.kneighbors(queries, return_distance=True)
    print(dist_sk,neighb_sk)

    #### bigsise's kneighbors
    print('\n---------bigsises kneighbors--------------')
    neigh_own = NearestNeighborsOwn(n_neighbors=4,metric='euclidean')
    neigh_own.fit(samples)
    dist_own, neighb_own = neigh_own.kneighbors(queries, return_distance=True)
    print(dist_own, neighb_own)

    ### Comparison
    print('\n--------np.allclose of the distance matrices------------')
    print(np.allclose(np.sort(neighb_sk), np.sort(neighb_own)))

    ###################################################################################################################
    #### Compare radius
    print("\n--------------- compare radius --------------------")

    #samples = np.random.rand(300, 100)
    #queries = np.random.rand(200, 100)
    samples = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5], [1, 1, 1]])
    queries = np.array([[1., 1., 1.], [0., .5, 0]])

    #### sklearn's radius_neighbors
    print('\n--------sklearns radius_neighbors--------------')
    neigh_radius = NearestNeighbors(radius=1.4,algorithm='brute')
    neigh_radius.fit(samples)
    dist_matrix, ind_matrix = neigh_radius.radius_neighbors(queries, sort_results=True)

    # 2Do: Print these sorted
    print(dist_matrix)
    print(ind_matrix)

    #### bigsises's radius_neighbors
    print('\n--------bigsises radius_neighbors--------------')
    neigh_radius_own = NearestNeighborsOwn(radius=1.4)
    neigh_radius_own.fit(samples)
    dist_matrix_own, ind_matrix_own = neigh_radius_own.radius_neighbors(queries, sort_results=True)

    # 2Do: Ditto
    print(dist_matrix_own)
    print(ind_matrix_own)

    ### Comparison
    # print('\n--------np.allclose of the distance matrices------------')
    # print(np.allclose(neigh.radius_neighbors(queries, return_distance=False), neigh_own.radius_neighbors(queries,
                                                                                             # return_distance=False)))

    ###################################################################################################################
    #### Compare KNeighborsClassifier
    print("-------------------------------------------------------------",
          "\n-------------- compare KNeighborsClassifier -----------------",
          "\n-------------------------------------------------------------")

    ### sklearn's KNeighborsClassifier's predict
    print('\n--------sklearns predict--------------')
    X = np.array([[0,0], [1,1], [2,2], [3,3]])

    #X = [[0], [1], [2], [3]]
    y = np.array([0, 0, 1, 1])
    #y = [[0, 'a'], [0, 'a'], [1, 'b'], [1,'b']]

    neigh = KNeighborsClassifier(n_neighbors=3,algorithm='brute')
    neigh.fit(X, y)

    print(neigh.predict(np.array([[1.1,1.1]])))
    print(neigh.predict_proba([[0.9,0.9]]))

    ### bigsise's KNeighborsClassifier's predict
    print('\n--------bigsises predict--------------')

    neigh_own = KNeighborsClassifierOwn(n_neighbors=3)
    neigh_own.fit(X, y)

    print(neigh_own.predict(np.array([[1.1, 1.1]])))
    print(neigh_own.predict_proba(np.array([[0.9, 0.9]])))


    ###################################################################################################################
    #### Compare KNeighborsRegressor
    print("-------------------------------------------------------------",
          "\n-------------- compare KNeighborsRegressor ------------------",
          "\n-------------------------------------------------------------")
    # X = [[0], [1], [2], [3]]
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    # y = [[0, 0], [0, 0], [1, 1], [1, 1]]

    print('\n--------sklearns predict--------------')
    regr_sk = KNeighborsRegressor(2,algorithm='brute')
    regr_sk.fit(X, y)
    print(regr_sk.predict(np.array([[1.5, 1.5]])))

    print('\n--------bigsises predict--------------')
    regr_bigsise = KNeighborsRegressorOwn(2)
    regr_bigsise.fit(X, y)
    print(regr_bigsise.predict(np.array([[1.5, 1.5]])))


    ###################################################################################################################
    #### Compare RadiusNeighborsRegressor
    print("-------------------------------------------------------------",
          "\n-------------- compare RadiusNeighborsRegressor -------------",
          "\n-------------------------------------------------------------")

    print('\n--------sklearns predict--------------')
    # samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
    #neigh = NearestNeighbors(radius=1.6)
    #neigh.fit(samples)
    #g = neigh.radius_neighbors([[1., 1., 1.]])
    #print(np.asarray(rng[0][0]))
    #print(np.asarray(rng[1][0]))

    X = np.array([[0], [1], [2], [3]])
    #X = [[0, 0], [1, 1], [2, 2], [3, 3]]
    y = np.array([0, 0, 1, 1])
    #y = [[0, 0], [0, 0], [1, 1], [1, 1]]

    rad_regr_sk = RadiusNeighborsRegressor(0.5,algorithm='brute')
    rad_regr_sk.fit(X, y)
    # print(rad_regr_sk.radius_neighbors(np.array([[1.5], [2.8]])))
    print(rad_regr_sk.predict(np.array([[1.5],[2.8]])))

    print('\n--------bigsises predict--------------')
    rad_regr_bigsise = RadiusNeighborsRegressorOwn(0.5)
    rad_regr_bigsise.fit(X, y)
    # print(rad_regr_bigsise.radius_neighbors(np.array([[1.5], [2.8]])))
    print(rad_regr_bigsise.predict(np.array([[1.5],[2.8]])))


    ###################################################################################################################
    #### Compare RadiusNeighborsClassifier
    print("-------------------------------------------------------------",
          "\n-------------- compare RadiusNeighborsClassifier -------------",
          "\n-------------------------------------------------------------")


    X = np.array([[0., 0., 0.], [0., .5, 0.], [1., 1., .5]])
    y = np.array([0, 1, 2])

    print('\n--------sklearns predict--------------')
    rad_classifier_sk = RadiusNeighborsClassifier(0.5,algorithm='brute')
    rad_classifier_sk.fit(X,y)
    print(rad_classifier_sk.predict([[0, 0.6, 0.0]]))
    print(rad_classifier_sk.predict_proba([[0, 0.6, 0.0]]))

    print('\n--------bigsises predict--------------')
    rad_classifier_bigsise = RadiusNeighborsClassifierOwn(0.5)
    rad_classifier_bigsise.fit(X,y)
    print(rad_classifier_bigsise.predict(np.array([[0, 0.6, 0.0]])))
    print(rad_classifier_bigsise.predict_proba(np.array([[0, 0.6, 0.0]])))

    ###################################################################################################################
    #### Compare KMeans
    print("-------------------------------------------------------------",
          "\n-------------- compare KMeans -------------",
          "\n-------------------------------------------------------------")

    C1 = np.random.multivariate_normal(np.array([0,0]),np.eye(2),size=100)
    C2 = np.random.multivariate_normal(np.array([1,1]),np.eye(2),size=100)

    X = np.vstack([C1,C2])

    permutation = np.arange(len(X))
    np.random.shuffle(permutation)

    X = X[permutation]

    init = X[:2]

    print('\n--------sklearns predict--------------')
    kmeans_sk = KMeans(n_clusters=2,n_init=1,init=init,random_state=47)
    kmeans_sk.fit(X)
    kmeans_sk.cluster_centers_ = np.sort(kmeans_sk.cluster_centers_,axis=0)
    print(kmeans_sk.cluster_centers_)
    print(kmeans_sk.predict([[0, 0.6]]))

    print('\n--------bigises predict--------------')
    kmeans_bigsise = MiniBatchKMeansOwn(n_clusters=2,init=init,random_state=47)
    kmeans_bigsise.fit(X)
    print(kmeans_bigsise.cluster_centers_)
    print(kmeans_bigsise.predict(np.array([[0, 0.6]])))
    #
    # ###################################################################################################################
    # #### Compare PCA
    print("-------------------------------------------------------------",
          "\n-------------- compare PCA -------------",
          "\n-------------------------------------------------------------")

    C1 = np.random.multivariate_normal(np.array([0,0]),np.eye(2),size=100)
    C2 = np.random.multivariate_normal(np.array([1,1]),np.eye(2),size=100)

    X = np.vstack([C1,C2])

    permutation = np.arange(len(X))
    np.random.shuffle(permutation)

    X = X[permutation]

    print('\n--------sklearns PCA--------------')
    pca_sk = PCA(n_components=2)
    # pca_sk_approx = PCA(n_components=2,svd_solver='randomized')
    pca_sk.fit(X)
    # pca_sk_approx.fit(X)
    print(pca_sk.components_)
    print(pca_sk.singular_values_)
    # print(kmeans_sk.predict([[0, 0.6]]))

    print('\n--------bigises PCA--------------')
    pca_bigsise = PCAOwn(n_components=2)
    pca_bigsise.fit(X)
    print(pca_bigsise.components_)
    print(pca_bigsise.singular_values_)
    #
    # ##### GROUND TRUTH #####
    # print('\n--------ground truth PCA--------------')
    # from scipy.linalg import svd
    # _,S,W_T = svd(X-X.mean(axis=0),full_matrices=True)
    # print(S,W_T.T)
    # ########################

    # plt.figure()
    # origin = np.array([[0,0],[0,0]])
    # plt.quiver(*origin, pca_sk.components_[:,0], pca_sk.components_[:,1], color=['r'], scale=10)
    # plt.quiver(*origin, V[:,0], V[:,1], color=['b'], scale=10)
    # plt.scatter(X[:,0],X[:,1])
    # plt.show()
