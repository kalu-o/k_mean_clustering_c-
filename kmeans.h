#ifndef KMEANS_H
#define KMEANS_H
#include "util.cpp"

/*
 *	Some simple, useful #define's
 */
#define EPSILON 1e-8
#define MAX_ERROR 1e8

/*
 *	The Kmeans class: for Kmeans clustering algorithm
 */
template <class T, class U>
class KMeans
{
    protected:
        long unsigned num_clusters;
        std::vector<ClusterNode<T>> clusters;
        ErrorFunction<T, U>* objective; //EuclideanDist<T, U>* p = dynamic_cast<EuclideanDist<T, U>*>(objective)
    public:
        KMeans(std::vector<ClusterNode<T>> in_data, ErrorFunction<T, U>* obj = new EuclideanDist<T, U>):
             num_clusters{in_data.size()}, clusters{in_data}, objective{obj}{}
        KMeans(long unsigned num_c = 10, ErrorFunction<T, U>* obj = new EuclideanDist<T, U>): num_clusters{num_c}, objective{obj}{}
        KMeans(const KMeans& arg): num_clusters{arg.num_clusters}, clusters{arg.clusters}{}
        long unsigned getNumClusters(){return num_clusters;}
        int setNumClusters(int n_clusters){num_clusters = n_clusters; return 0;}
        std::vector<ClusterNode<T>>& getClusters(){return clusters;}
        int setClusters(int _clusters){clusters = _clusters; return 0;}
        const KMeans& operator=(const KMeans&);
        VectorNodes<T, U>& readCSV(std::string, VectorNodes<T, U>&);
        KMeans<T, U>& initializeCentroid(VectorNodes<T, U>&);
        KMeans<T, U>& initializeCentroid(std::vector<ClusterNode<T>>&);
        int assignCluster(PointNode<T>&);
        int updateCentroid(VectorNodes<T, U>&);
        int kmeansAlgorithm(VectorNodes<T, U>&);
        int describeCluster();
        ~ KMeans(){}
};


#endif
