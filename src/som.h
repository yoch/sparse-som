#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED


#include "data.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <cfloat>
#include <cmath>
#include <chrono>

// helper
template<class T>
inline T squared(T x)
{
    return x * x;
}

inline double get_wall_time()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


#define HEX_H 0.8660254037844386

inline double eucdist (int y, int x, int i, int j, double & d2)
{
    d2 = squared(i-y) + squared(j-x);
    return std::sqrt(d2);
}

inline double eucdist_hexa(int y, int x, int i, int j, double & d2)
{
    double x6 = (x&1) ? 0.5 + x : x;
    double j6 = (j&1) ? 0.5 + j : j;
    d2 = squared((i-y)*HEX_H) + squared(j6-x6);
    return std::sqrt(d2);
}

inline double manhatan_dist(int y, int x, int i, int j, double & d2)
{
    d2 = squared(i-y) + squared(j-x); // regular d^2
    return std::max(std::abs(i-y),std::abs(j-x));
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

    Som(size_t, size_t, size_t, topology=RECT, int=0);
    Som(const std::string& filename, topology=RECT, int=0);
    ~Som();

    void train(const CSR&, size_t tmax,
               double r0, double a0, double rN=FLT_MIN, double aN=FLT_MIN,
               double stdCoeff=0.3, cooling rc=LINEAR, cooling ac=LINEAR);

    std::vector<bmu> getBmus(const CSR&) const;
    std::vector<label_counter> calibrate(const dataset& dataSet) const;

    // IO gestion
    void dump(const std::string& filename, bool binary=false) const;
    void dumpSparse(const std::string& filename, double=FLT_EPSILON) const;
    void classify(const dataset&, const std::string& filename, bool cnt) const;

    inline size_t getx() const {return m_width;}
    inline size_t gety() const {return m_height;}
    inline size_t getz() const {return m_dim;}
    inline double* _codebookptr() const {return codebook;}

    inline int getverb() const {return m_verbose;}
    inline void setverb(int v) {m_verbose = v;}


private:

    void init();
    void update(const CSR& data, size_t n, size_t k, double r, double a, double s);
    size_t getBmu(const CSR& data, const size_t n, double& d) const;
    void stabilize(size_t k);

    /// attributes

    size_t m_height;      // lig x col = nombre de neurones de la carte
    size_t m_width;
    size_t m_dim;  // taille du vecteur

    topology m_topo;
	int m_verbose;
    double (*fdist) (int, int, int, int, double&);

    double* codebook;
    double* squared_sum;
    double* w_coeff;
    double * wvprod;
};


class BSom
{

public:

    BSom(size_t, size_t, size_t, topology=RECT, int=0);
    BSom(const std::string& filename, topology=RECT, int=0);
    ~BSom();

    void train(const CSR&, size_t tcoef,
               float r0, float rN=0.f, float stdCoef=0.3, cooling rc=LINEAR);

    void getBmus(const CSR&, size_t * const bmus, float * const dsts, size_t * const second=NULL, float * const sdsts=NULL) const;
    std::vector<label_counter> calibrate(const dataset& dataSet) const;

    // IO gestion
    void dump(const std::string& filename, bool binary=false) const;
    void dumpSparse(const std::string& filename, double=FLT_EPSILON) const;
    void classify(const dataset&, const std::string& filename, bool cnt) const;

    inline size_t getx() const {return m_width;}
    inline size_t gety() const {return m_height;}
    inline size_t getz() const {return m_dim;}
    inline float* _codebookptr() const {return codebook;}

    inline int getverb() const {return m_verbose;}
    inline void setverb(int v) {m_verbose = v;}

private:

    void init();
    void update(const CSR& data, const float r, const float s, size_t * const bmus);
    //size_t getBmu(const sparse_vec& v, float& d) const;
    void trainOneEpoch(const CSR&, size_t t, size_t tmax,
                       float radius0, float radiusN, float stdCoef, cooling rc,
                       size_t * const bmus, float * const dsts);

    /// attributes

    size_t m_height;      // lig x col = nombre de neurones de la carte
    size_t m_width;
    size_t m_dim;  // taille du vecteur

    topology m_topo;
    int m_verbose;
    double (*fdist) (int, int, int, int, double&);

    float* codebook;
};

}

#endif // SOM_H_INCLUDED
