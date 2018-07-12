#include "kmeans.h"
#include <typeinfo>


/*
 *	The readCSV method reads a comma separated
 *  input file into a VectorNodes data structure
 */

template <class T, class U>
VectorNodes<T, U>& KMeans<T, U>::readCSV(std::string file_name, VectorNodes<T, U>& input)
{
    std::string csv_line;
    std::fstream file(file_name, std::ios::in);
    if(!file.is_open())
    {
        std::cout << "File not found!\n";
        EXIT_FAILURE;
    }
    // read every line from the stream
    while( getline(file, csv_line) )
    {
        std::istringstream csv_stream(csv_line);
        std::vector<T> data;
        std::string csv_element;
        // read every element from the line that is separated by commas
        while(getline(csv_stream, csv_element, ',') )
        {
            data.push_back(atof(csv_element.c_str()));
        }
        input.insertNode(new PointNode<T>(data));
        data.clear();
    }
    return input;
}
/*
 *	This initializes centroids of a KMeans
 *  clusters randomly from the train data
 */
template <class T, class U>
KMeans<T, U>& KMeans<T, U>::initializeCentroid(VectorNodes<T, U>& input)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    unsigned _min = 0;
    unsigned _max = input.getNodeCount() - 1;
    auto r_integer = 0;
    ClusterNode<T>* p_cluster;
    std::vector<T> data;
    std::set<T> temp;
    for (unsigned i = 0; i < this->num_clusters; ++i)
    {
        std::uniform_int_distribution<int> uni(_min, _max);
        r_integer = uni(rng);
        while (temp.find(r_integer) != temp.end()) r_integer = uni(rng);
        data = input.getNodes()[r_integer].getData();
        p_cluster = new ClusterNode<T>(data);
        p_cluster->setClusterId(i);
        p_cluster->setNumNodes(0);
        clusters.push_back(*p_cluster);
        temp.insert(r_integer);
    }
    return *this;
}
/*
 *	This initializes centroids of a KMeans
 *  clusters from existing centroids
 */
template <class T, class U>
KMeans<T, U>& KMeans<T, U>::initializeCentroid(std::vector<ClusterNode<T>>& clusters)
{
    this->clusters = clusters;
    return *this;
}
/*
 *	Overloaded assignment operator of the KMeans class
 */
template <class T, class U>
const KMeans<T, U>& KMeans<T, U>::operator=(const KMeans& arg)
{
    if(this != &arg)
    {
        this->num_clusters = arg.num_clusters;
        this->clusters     = arg.clusters;
        this->objective    = new EuclideanDist<T, U>();
        *objective = *arg.objective;
    }
    return *this;
}
/*
 *	Assigns a point to a cluster based on the Euclidean distance
 */
template <class T, class U>
int KMeans<T, U>::assignCluster(PointNode<T>& in_node)
{
    unsigned index_min_dist = 0;
    float distance = 0, min_distance = MAX_ERROR;
    EuclideanDist<T, U>* p_dist = dynamic_cast<EuclideanDist<T, U>*>(objective);
    std::vector<T> data, centroid;

    for (unsigned i = 0; i < num_clusters; ++i)
    {
        data = in_node.getData();
        centroid = clusters[i].getData();
        p_dist->costFunction(data, centroid, distance);
        if (distance < min_distance)
        {
            index_min_dist = i;
            min_distance = distance;
        }
    }
    in_node.setClusterId(index_min_dist);
    in_node.setNodeError(min_distance);
    return 0;
}
/*
 *	Updates cluster centroid(mean)
 */
template <class T, class U>
int KMeans<T, U>::updateCentroid(VectorNodes<T, U>& in_nodes)
{
    std::vector<Node<T>*> p_temp;
    std::vector<unsigned> num_nodes(num_clusters, 0);
    unsigned cluster_id, data_size = clusters[0].getDataSize();
    float total_error = 0;
    for (unsigned i = 0; i < num_clusters; ++i) p_temp.push_back(new Node<T>(data_size));
    for (auto& n: in_nodes.getNodes())
    {
        cluster_id = n.getClusterId();
        *p_temp[cluster_id] = *p_temp[cluster_id] + n;
        num_nodes[cluster_id] += 1;
        total_error += n.getNodeError();
    }

    for (unsigned i = 0; i < num_clusters; ++i)
    {
        p_temp[i]->scalarMul(1.0/num_nodes[i]);
        clusters[i].setData(p_temp[i]->getData());
    }
    in_nodes.setTotalError(total_error);
    return 0;
}
/*
 *	The k-means clustering algorithm.
 */
template <class T, class U>
int KMeans<T, U>::kmeansAlgorithm(VectorNodes<T, U>& in_nodes)
{
    std::vector<U> temp = in_nodes.getNodes();
    /*
     * Expectation step.
     */
    for (auto& n: temp)assignCluster(n);
    in_nodes.setNodes(temp);
    /*
     * Maximization step.
     */
    updateCentroid(in_nodes);
    return 0;
}
/*
 *	Prints cluster centroids to the console.
 */
template <class T, class U>
int KMeans<T, U>::describeCluster()
{
    std::cout << "clusters: "<< num_clusters <<std::endl;
    for (unsigned i = 0; i < num_clusters; ++i)clusters[i].describeNode();
    return 0;
}
