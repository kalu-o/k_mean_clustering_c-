#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

/*
 * A wrapper around the vector class. Provides a container for objects
*/
template <class T, class U>
class VectorNodes
{
    private:
        std::vector<U> nodes;
        int node_count;
        float total_error;
    public:
        VectorNodes(): node_count{0}, total_error{0}{}
        std::vector<U>& getNodes(){return nodes;}
        int setNodes(std::vector<U>& _nodes){nodes = _nodes; return 0;}
        int getNodeCount(){node_count = nodes.size(); return node_count;}
        int setNodeCount(int _count){node_count = _count; return 0;}
        float getTotalError(){return total_error;}
        int setTotalError(float t_error){total_error = t_error; return 0;}
        int insertNode(U* node){nodes.push_back(*node); return 0;};
        int clearVector(){nodes.clear(); node_count = 0; return 0;}
        int describeList();
        ~ VectorNodes(){}
};

/*
 *This is the Node class, the foundation data structure used in the List class
*/
template <class T>
class Node
{
    protected:
        std::vector<T> data;
        long unsigned data_size;

    public:
        Node():data_size{0}{}
        Node(std::vector<T> in_data): data{in_data}, data_size{in_data.size()}{}
        Node(long unsigned data_size, int output_dim){generate_data(data_size, output_dim);}
        Node(long unsigned data_size, int output_dim, int init):  data{*new std::vector<T>(data_size, init)}, data_size{data_size} {}
        Node(long unsigned data_size): data{*new std::vector<T>(data_size, 0)}, data_size{data_size}  {}
        Node(const Node<T>& in_node): data{in_node.data}, data_size{in_node.data_size}{}
        std::vector<T> getData ()const{return data;}
        int setData(std::vector<T> in_data){data = in_data; return 0;}
        int getDataSize(){return data_size;}
        int setdataSize(int data_size){this->data_size = data_size; return 0;}
        Node<T>& operator+(const Node<T> &);
        Node<T>& operator-(const Node<T> &);
        Node<T>& operator*(const Node<T> &);
        const Node& operator=(const Node&);
        T sum();
        int sigmoid();
        int sigmoidPrime();
        int generate_data(int, int);
        int describeNode();
        Node<T>& scalarMul(float);
        virtual ~ Node(){}
};
/*
 *the ListNode class derives from the Node class
*/
template <class T>
class ListNode: public Node <T>
{
    private:
        ListNode *p_next_node, *p_prev_node;
        int index;
    public:
        ListNode(): p_next_node{nullptr}, p_prev_node{nullptr}, index{0}{}
        ListNode(std::vector<T> in_data): Node<T>(in_data), p_next_node{nullptr}, p_prev_node{nullptr}, index{0}{}
        ListNode(int data_size, int output_dim): Node<T>(data_size, output_dim), p_next_node{nullptr}, p_prev_node{nullptr}, index{0}{}
        ListNode(int data_size, int output_dim, int init): Node<T>(data_size, output_dim, init), p_next_node{nullptr}, p_prev_node{nullptr}, index{0}{}
        ListNode(const ListNode<T>& in_node): Node<T>(in_node), p_next_node{in_node.p_next_node}, p_prev_node{in_node.p_prev_node}, index{in_node.index} {}
        ListNode* getPtrNext(){return p_next_node;}
        int setPtrNext(ListNode *ptr){p_next_node = ptr; return 0;}
        ListNode* getPtrPrev(){return p_prev_node;}
        int setPtrPrev(ListNode *ptr){p_prev_node = ptr; return 0;}
        int getIndex(){return index;}
        int setIndex(int ind){index = ind; return 0;}
        const ListNode& operator=(const ListNode&);
        ListNode<T>& scalarMul(float);
        ~ ListNode(){p_next_node = nullptr; p_prev_node = nullptr; }
};
/*
 * The PointNode class derives from the Node class. Used as a data structure
 * for storing multi-dimensional data.
*/
template <class T>
class PointNode: public Node <T>
{
    private:
        PointNode *p_next_node, *p_prev_node;
        int index, cluster_id;
        float node_error;
    public:
        PointNode(): p_next_node{nullptr}, p_prev_node{nullptr}, index{0}, cluster_id{0}, node_error{0}{}
        PointNode(std::vector<T> in_data): Node<T>(in_data), p_next_node{nullptr}, p_prev_node{nullptr}, index{0}, cluster_id{0}, node_error{0}{}
        PointNode(const PointNode<T>& in_node): Node<T>(in_node), p_next_node{nullptr}, p_prev_node{nullptr}, index{in_node.index},
        cluster_id{in_node.cluster_id}, node_error{in_node.node_error}{}
        PointNode* getPtrNext(){return p_next_node;}
        int setPtrNext(PointNode *ptr){p_next_node = ptr; return 0;}
        PointNode* getPtrPrev(){return p_prev_node;}
        int setPtrPrev(PointNode *ptr){p_prev_node = ptr; return 0;}
        int getIndex(){return index;}
        int setIndex(int ind){index = ind; return 0;}
        int getClusterId(){return cluster_id;}
        int setClusterId(int id){cluster_id = id; return 0;}
        float getNodeError(){return node_error;}
        int setNodeError(float error){node_error = error; return 0;}
        const PointNode& operator=(const PointNode&);
        PointNode<T>& scalarMul(float);
        ~ PointNode(){p_next_node = nullptr; p_prev_node = nullptr; }
};
/*
 * The ClusterNode class derives from the Node class. Used as a data structure
 * for storing centroids in used in Kmeans clustering.
*/
template <class T>
class ClusterNode: public Node<T>
{
    private:
        int cluster_id, num_nodes;
    public:
        ClusterNode(): Node<T>(), cluster_id{0}, num_nodes{0}{}
        ClusterNode(std::vector<T> in_data): Node<T>(in_data), cluster_id{0}, num_nodes{0}{}
        ClusterNode(const ClusterNode<T>& in_node): Node<T>(in_node), cluster_id{in_node.cluster_id}, num_nodes{in_node.num_nodes}{}
        int getClusterId(){return cluster_id;}
        int setClusterId(int id){cluster_id = id; return 0;}
        int getNumNodes(){return num_nodes;}
        int setNumNodes(int n_nodes){num_nodes = n_nodes; return 0;}
        const ClusterNode& operator=(const ClusterNode&);
        ~ ClusterNode(){}
};

/*
 *the List class is an abstract class
*/
template <class T>
class List
{
    public:
        int insertNode(ListNode<T>*, int pos = -1);
        //virtual int insertNode(ListNode<T>*, int pos = -1) = 0;
        virtual int deleteNode(int pos = -1) = 0;
        //virtual int sumList(T &) = 0;
        //virtual int sigmoidList() = 0;
        //virtual int sigmoidPrimeList() = 0;
        virtual int describeList() = 0;
        virtual ~ List() {};

};
/*
 *the linkedList class derives from the List class
*/
template <class T>
class LinkedList: public List<T>
{
    private:
        ListNode<T> *p_first_node, *p_last_node;
        int node_count, output_dim, input_dim;

    public:
        LinkedList(): p_first_node{nullptr}, p_last_node{nullptr}, node_count{0}, input_dim{0}, output_dim{0}{}
        LinkedList(int, int);
        LinkedList(int, int, int);
        ListNode<T>* getPtrFirst(){ return p_first_node;}
        ListNode<T>* getPtrLast(){ return p_last_node;}
        int setPtrFirst(ListNode<T>* p_first_node){this->p_first_node = p_first_node; return 0;}
        int setPtrLast(ListNode<T>* p_last_node){this->p_last_node = p_last_node; return 0;}
        virtual int insertNode(ListNode<T> *, int pos = -1);
        virtual int deleteNode(int pos = -1);
        virtual int sumList(T &);
        virtual int sigmoidList();
        virtual int sigmoidPrimeList();
        LinkedList<T>& dot2(const LinkedList<T> &);
        int dot(ListNode<T> &, ListNode<T> &);
        int describeList();
        LinkedList<T>& operator+(const LinkedList<T> &);
        LinkedList<T>& operator-(const LinkedList<T> &);
        LinkedList<T>& multiply(ListNode<T>&, ListNode<T>&);
        LinkedList<T>& transpose(LinkedList<T>&);
        int setDataZero();
        LinkedList<T>& scalarMulList(float);
        LinkedList<T>& clearList();
        ~ LinkedList(){}
};
/*
 *the ClusterList class derives from the List class
*/
template <class T>
class ClusterList: public List<T>
{
    private:
        PointNode<T> *p_first_node, *p_last_node;
        int node_count;
        float total_error;

    public:
        ClusterList(): p_first_node{nullptr}, p_last_node{nullptr}, node_count{0}, total_error{0}{}
        //ClusterList(int, int);
        PointNode<T>* getPtrFirst(){ return p_first_node;}
        PointNode<T>* getPtrLast(){ return p_last_node;}
        int setPtrFirst(PointNode<T>* p_first_node){this->p_first_node = p_first_node; return 0;}
        int setPtrLast(PointNode<T>* p_last_node){this->p_last_node = p_last_node; return 0;}

        int getNodeCount(){return node_count;}
        int setNodeCount(int _count){node_count = _count; return 0;}

        float getTotalError(){return total_error;}
        int setTotalError(float error){total_error = error; return 0;}

        int insertNode(PointNode<T> *, int pos = -1);
        int deleteNode(int pos = -1);
        int describeList();
        ClusterList<T>& clearList();
        ClusterList<T>& readCSV(std::string);
        ~ ClusterList(){}
        /*
        ClusterList<T>& operator+(const ClusterList<T> &);
        ClusterList<T>& operator-(const ClusterList<T> &);
        ClusterList<T>& multiply(ClusterNode<T>&, ClusterNode<T>&);
        ClusterList<T>& transpose(ClusterList<T>&);
        int setDataZero();
        ClusterList<T>& scalarMulList(float);*/
};

/*
 *an abstract class Objective: responsible for objective function
*/
template <class T>
class Objective
{
    public:
        virtual int costFunction(LinkedList<T> &, LinkedList<T> &, T &) = 0;
        virtual ListNode<T> costDerivative(ListNode<T>&, ListNode<T>&, ListNode<T>&) = 0;

};
/*
 *the MSE (mean squared error cost function) derives from the class Objective
*/
template <class T>
class MSE: public Objective<T>
{
    public:
        int costFunction(LinkedList<T> &predictions, LinkedList<T> &labels, T &result);
        ListNode<T> costDerivative(ListNode<T> &predictions, ListNode<T> &labels, ListNode<T> &result);

};
/*
 *an abstract class ErrorFunction: responsible for cost/error function
*/
template <class T, class U>
class ErrorFunction
{
    public:
        virtual int costFunction(std::vector<T>&, std::vector<T>&, T &) = 0;
        virtual int totalCost(VectorNodes<T, U>&, T&) = 0;

};
/*
 * The EuclideanDist derives from the class ErrorFunction
 * Computes the Euclidean distance between two points.
*/
template <class T, class U>
class EuclideanDist: public ErrorFunction<T, U>
{
    public:
        int costFunction(std::vector<T>&, std::vector<T>&, T &);
        int totalCost(VectorNodes<T, U>&, T&);

};
#endif


