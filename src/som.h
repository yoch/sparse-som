#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED


#include "data.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <cfloat>

// helper
template<class T>
inline T squared(T x)
{
    return x * x;
}


namespace som {

struct bmu
{
    bmu(size_t i, double d) :
        idx(i), dst(d) {}
    size_t idx;
    double dst;
};

enum topology
{
    RECT=8, HEXA=6, CIRC=4
};

enum cooling
{
    LINEAR=0, EXPONENTIAL=1
};


using label_counter = std::unordered_map<std::string, size_t>;

class Som
{

public:

    Som(size_t, size_t, size_t, topology=RECT, bool=false);
    Som(const std::string& filename, topology=RECT, bool=false);
    ~Som();

    void train(const std::vector<sparse_vec>&, size_t tmax,
               double r0, double a0, double rN=FLT_MIN, double aN=FLT_MIN,
               double stdCoeff=0.3, cooling rc=LINEAR, cooling ac=LINEAR);

    std::vector<bmu> getBmus(const std::vector<sparse_vec>&) const;
    std::vector<label_counter> calibrate(const dataset& dataSet) const;

    // IO gestion
    void dump(const std::string& filename, bool binary=false) const;
    void dumpSparse(const std::string& filename, double=FLT_EPSILON) const;
    void classify(const dataset&, const std::string& filename, bool cnt) const;

    inline size_t getx() const {return m_width;}
    inline size_t gety() const {return m_height;}
    inline size_t getz() const {return m_dim;}
    inline double* _codebookptr() const {return codebook;}


private:

    void init();
    void update(const sparse_vec& v, size_t k, double r, double a, double s);
    size_t getBmu(const sparse_vec& v, double& d) const;
    void stabilize(size_t k);

    /// attributes

    size_t m_height;      // lig x col = nombre de neurones de la carte
    size_t m_width;
    size_t m_dim;  // taille du vecteur

    topology m_topo;
	bool m_verbose;

    double* codebook;
    double* squared_sum;
    double* w_coeff;
    double * wvprod;
};


class BSom
{

public:

    BSom(size_t, size_t, size_t, topology=RECT, bool=false);
    BSom(const std::string& filename, topology=RECT, bool=false);
    ~BSom();

    void train(const std::vector<sparse_vec>&, size_t tcoef,
               float r0, float rN=0.f, float stdCoef=0.3, cooling rc=LINEAR);

    void getBmus(const std::vector<sparse_vec>&, size_t * const bmus, float * const dsts) const;
    std::vector<label_counter> calibrate(const dataset& dataSet) const;

    // IO gestion
    void dump(const std::string& filename, bool binary=false) const;
    void dumpSparse(const std::string& filename, double=FLT_EPSILON) const;
    void classify(const dataset&, const std::string& filename, bool cnt) const;

    inline size_t getx() const {return m_width;}
    inline size_t gety() const {return m_height;}
    inline size_t getz() const {return m_dim;}
    inline float* _codebookptr() const {return codebook;}


private:

    void init();
    void update(const std::vector<sparse_vec>& data, const float r, const float s, size_t * const bmus);
    //size_t getBmu(const sparse_vec& v, float& d) const;
    void trainOneEpoch(const std::vector<sparse_vec>&, size_t t, size_t tmax,
                       float radius0, float radiusN, float stdCoef, cooling rc,
                       size_t * const bmus, float * const dsts);

    /// attributes

    size_t m_height;      // lig x col = nombre de neurones de la carte
    size_t m_width;
    size_t m_dim;  // taille du vecteur

    topology m_topo;
    bool m_verbose;

    float* codebook;
};

}

#endif // SOM_H_INCLUDED
