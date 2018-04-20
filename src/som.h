#ifndef SOM_H_INCLUDED
#define SOM_H_INCLUDED


#include "data.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <cfloat>
#include <limits>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional>
#include <chrono>

#if defined(_OPENMP)
    #include <omp.h>
#endif


#if defined(_MSC_VER)
    using idx_t = int;
#else
    using idx_t = unsigned int;
#endif


inline double get_wall_time()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


////////////////////////////////////////////// Math Helpers //////////////////////////////////////////////

template<class T>
inline T squared(T x)
{
    return x * x;
}

template<class T>
static T prod(const T * const w, const float * const vsp, const int * const ind, const size_t vsz)
{
    T ret = 0.;
    for (size_t it=0; it<vsz; ++it)
    {
        ret += vsp[it] * w[ind[it]];
    }
    return ret;
}

template<class T>
static T squared(const T * const w, size_t sz)
{
    T ret = 0.;
    for (size_t i=0; i < sz; ++i)
    {
        ret += squared(w[i]);
    }
    return ret;
}


////////////////////////////// IO Helpers ///////////////////////////////////

#define BUFSIZE 16777216

// IO buffer (64 Mo)
//extern 
static float iobuf[BUFSIZE];

template<typename RealType>
static inline void read_raw_codebook_f32(std::ifstream& myfile, RealType* codebook, size_t sz)
{
    for (size_t k=0; k < sz; k+=BUFSIZE)
    {
        size_t toread = std::min<size_t>(BUFSIZE, sz - k);
        myfile.read((char*)iobuf, sz * sizeof(*iobuf));
        std::copy(iobuf, iobuf + toread, codebook + k);
    }
}

template<typename RealType>
static inline void write_raw_codebook_f32(std::ofstream& myfile, RealType* codebook, size_t sz)
{
    for (size_t k=0; k < sz; k += BUFSIZE)
    {
        size_t towrite = std::min<size_t>(BUFSIZE, sz - k);
        std::copy(codebook + k, codebook + k + towrite, iobuf);
        myfile.write((const char*)iobuf, towrite * sizeof(*iobuf));
    }
}

static inline void read_raw_codebook_f32(std::ifstream& myfile, float* codebook, size_t sz)
{
    myfile.read((char*)codebook, sz * sizeof(*codebook));
}

static inline void write_raw_codebook_f32(std::ofstream& myfile, float* codebook, size_t sz)
{
    myfile.write((const char*) codebook, sz * sizeof(*codebook));
}

/*
inline void read_raw_codebook_f32(std::ifstream& myfile, float* codebook, size_t sz);
inline void write_raw_codebook_f32(std::ofstream& myfile, float* codebook, size_t sz);
*/


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
    CIRC_TOR=1, RECT_TOR=2, RECT=8, HEXA=6, CIRC=4
};

enum cooling
{
    LINEAR=0, EXPONENTIAL=1
};


using label_counter = std::unordered_map<std::string, size_t>;

/*
template<typename RealType>
void read_raw_codebook_f32(std::ifstream& myfile, RealType* codebook, size_t sz);
void read_raw_codebook_f32(std::ifstream& myfile, float* codebook, size_t sz);

template<typename RealType>
void write_raw_codebook_f32(std::ofstream& myfile, RealType* codebook, size_t sz);
void write_raw_codebook_f32(std::ofstream& myfile, float* codebook, size_t sz);
*/

template<typename RealType>
class _SomBase
{
public:
    _SomBase(size_t, size_t, size_t, topology=RECT, int=0);
    _SomBase(const std::string&, topology=RECT, int=0);
    virtual ~_SomBase() = 0;    // make it abstract

    void randinit();
    void dump(const std::string&, bool=false) const;
    void getBmus(const CSR&, size_t*, RealType*, size_t* se=NULL, RealType* sd=NULL) const;
    double topographicError(size_t*, size_t*, size_t) const;
    std::vector<label_counter> calibrate(const dataset&) const;
    void classify(const dataset&, const std::string&, bool) const;

    inline size_t getx() const {return m_width;}
    inline size_t gety() const {return m_height;}
    inline size_t getz() const {return m_dim;}
    inline RealType* _codebookptr() const {return codebook;}

    inline int getverb() const {return m_verbose;}
    inline void setverb(int v) {m_verbose = v;}

protected:
    void setTopology(topology);

    //double eucdist (int y, int x, int i, int j, double & d2) const;
    //double eucdist_hexa (int y, int x, int i, int j, double & d2) const;
    //double rectdist (int y, int x, int i, int j, double & d2) const;
    double eucdist_toroidal (int y0, int x0, int y1, int x1, double & d2) const;
    double rectdist_toroidal (int y0, int x0, int y1, int x1, double & d2) const;

    size_t m_height;      // lig x col = nombre de neurones de la carte
    size_t m_width;
    size_t m_dim;  // taille du vecteur

    topology m_topo;
	int m_verbose;
    std::function<double(int, int, int, int, double&)> fdist;
    //double (*fdist) (int, int, int, int, double&);

    RealType* codebook;
};


//////////////////////////////////// Distance Functions //////////////////////////////////////

#define HEX_H 0.8660254037844386

inline double eucdist (int y, int x, int i, int j, double & d2)
{
    d2 = squared(i-y) + squared(j-x);
    return std::sqrt(d2);
}

inline double eucdist_hexa (int y, int x, int i, int j, double & d2)
{
    double x6 = (x&1) ? 0.5 + x : x;
    double j6 = (j&1) ? 0.5 + j : j;
    d2 = squared((i-y)*HEX_H) + squared(j6-x6);
    return std::sqrt(d2);
}

inline double rectdist (int y, int x, int i, int j, double & d2)
{
    d2 = squared(i-y) + squared(j-x); // regular d^2
    return std::max(std::abs(i-y), std::abs(j-x));
}

template <class T>
double _SomBase<T>::eucdist_toroidal (int y0, int x0, int y1, int x1, double & d2) const
{
    if (y0 > y1) std::swap(y0, y1);
    if (x0 > x1) std::swap(x0, x1);
    const int dy = std::min(y1 - y0, y0 + (int)m_height - y1);
    const int dx = std::min(x1 - x0, x0 + (int)m_width - x1);
    d2 = squared(dy) + squared(dx);
    return std::sqrt(d2);
}

template <class T>
double _SomBase<T>::rectdist_toroidal (int y0, int x0, int y1, int x1, double & d2) const
{
    if (y0 > y1) std::swap(y0, y1);
    if (x0 > x1) std::swap(x0, x1);
    const int dy = std::min(y1 - y0, y0 + (int)m_height - y1);
    const int dx = std::min(x1 - x0, x0 + (int)m_width - x1);
    d2 = squared(dy) + squared(dx); // regular d^2
    return std::max(dy, dx);
}

template <class T>
_SomBase<T>::_SomBase(size_t h, size_t w, size_t d, topology topo, int verbose) :
    m_height(h),
    m_width(w),
    m_dim(d),
    m_topo(topo),
    m_verbose(verbose)
{
    setTopology(topo);

    codebook = new T[h*w*d];

    // randomly init values
    randinit();
}

template <class T>
_SomBase<T>::_SomBase(const std::string& filename, topology topo, int verbose) :
    m_topo(topo),
    m_verbose(verbose)
{
    setTopology(topo);

    std::ifstream myfile;
    myfile.open(filename, std::ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    if (m_verbose > 0)
    {
        std::cout << "loading codebook...";
        std::cout.flush();
    }

    std::string header;
    std::getline(myfile, header);

    std::istringstream iss(header);
    iss >> m_height >> m_width >> m_dim;

    if (iss.fail())
        throw "bad header in " + filename;

    const size_t sz = m_height * m_width * m_dim;

    // allocate memory
    codebook = new T[sz];

    read_raw_codebook_f32(myfile, codebook, sz);

    if (myfile.fail())
        throw "bad header in " + filename;

    myfile.close();

    if (m_verbose > 0)
    {
        std::cout << " OK" << std::endl;
        if (m_verbose > 1)
        {
            std::cout << "  dimensions: " << m_height << " x " << m_width << " x " << m_dim << std::endl;
        }
    }
}

template <class T>
_SomBase<T>::~_SomBase()
{
    delete [] codebook;
}

template <class T>
void _SomBase<T>::setTopology(topology topo)
{
    m_topo = topo;

    switch (m_topo)
    {
    case CIRC_TOR:
        fdist = [this](int y0, int x0, int y1, int x1, double & d2) { 
                    return this->eucdist_toroidal(y0, x0, y1, x1, d2);
                };
        break;
    case RECT_TOR:
        fdist = [this](int y0, int x0, int y1, int x1, double & d2) { 
                    return this->rectdist_toroidal(y0, x0, y1, x1, d2);
                };
        break;
    case CIRC:
        fdist = eucdist;
        break;
    case HEXA:
        fdist = eucdist_hexa;
        break;
    case RECT:
    default:
        fdist = rectdist;
        break;
    }
}

template <class T>
void _SomBase<T>::randinit()
{
    unsigned seed = std::time(NULL);
#pragma omp parallel firstprivate(seed)
  {
#if defined(_OPENMP)
    // set an unique seed for each thread
    seed += omp_get_thread_num();
#endif
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<T> dist(0., 1.);

#pragma omp for
    for (idx_t k=0; k < m_height*m_width*m_dim; ++k)
    {
        codebook[k] = dist(rng);
    }
  }
}

template <class T>
void _SomBase<T>::dump(const std::string& filename, bool binary) const
{
    std::ofstream myfile;
    myfile.open(filename, std::ios::binary);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    myfile << m_height << ' ' << m_width << ' ' << m_dim << '\n';

    if (binary)
    {
        write_raw_codebook_f32(myfile, codebook, m_height * m_width * m_dim);
    }
    else
    {
        for (size_t k=0; k < m_height*m_width; ++k)
        {
            T * w = codebook + k * m_dim;
            for (size_t i=0; i < m_dim; ++i)
                myfile << w[i] << ' ';
            myfile << '\n';
        }
    }

    myfile.close();
}

template <class T>
void _SomBase<T>::getBmus(const CSR& data, size_t * bmus, T * dsts, size_t * second, T * sdsts) const
{
    std::fill_n(bmus, data.nrows, 0);
    std::fill_n(dsts, data.nrows, std::numeric_limits<T>::max());

    if (second)
    {
        std::fill_n(second, data.nrows, 0);
        std::fill_n(sdsts, data.nrows, std::numeric_limits<T>::max());
    }

    for (size_t k=0; k < m_height * m_width; ++k)
    {
        const T * const w = codebook + k * m_dim;

        // precompute {w_k}^2
        const T w2 = squared(w, m_dim);

// ensure that two threads don't access shared data (at i index) at a time
#pragma omp parallel for // shared(data,k,w2)

        for (idx_t i=0; i < (idx_t) data.nrows; ++i)
        {
            const int ind = data.indptr[i];
            const int vsz = data.indptr[i+1] - ind;
            const float * const vsp = &data.data[ind];
            const int * const vind = &data.indices[ind];

            // pseudo squared distance, d_i = d_i + X_i^2
            const T dst = w2 - 2 * prod(w, vsp, vind, vsz);

            if (!second)
            {
                if (dst < dsts[i])
                {
                    bmus[i] = k;
                    dsts[i] = dst;
                }
            }
            else // second!=NULL
            {
                if (dst < dsts[i])
                {
                    second[i] = bmus[i];
                    sdsts[i] = dsts[i];
                    bmus[i] = k;
                    dsts[i] = dst;
                }
                else if (dst < sdsts[i])
                {
                    second[i] = k;
                    sdsts[i] = dst;
                }
            }
        }
    }
}

template <class T>
double _SomBase<T>::topographicError(size_t * bmus, size_t * second, size_t n) const
{
    size_t errors = 0;
    for (size_t k=0; k<n; ++k)
    {
        double d2;
        const int y = bmus[k] / m_width,
                  x = bmus[k] % m_width,
                  i = second[k] / m_width,
                  j = second[k] % m_width;
        if (fdist(y, x, i, j, d2) > 1)
        {
            errors++;
        }
    }

    return (double) errors / n;
}

template <class T>
std::vector<label_counter> _SomBase<T>::calibrate(const dataset& dataSet) const
{
    size_t * bmus = new size_t[dataSet.nrows];
    T * dsts = new T[dataSet.nrows];

    getBmus(dataSet, bmus, dsts);

    std::vector<label_counter> mappings = std::vector<label_counter>(m_height*m_width);
    for (int k=0; k < dataSet.nrows; ++k)
    {
        const size_t kStar = bmus[k];
        mappings[kStar][dataSet.labels[k]]++;
    }

    delete [] bmus;
    delete [] dsts;

    return mappings;
}

template <class T>
void _SomBase<T>::classify(const dataset& dataSet, const std::string& filename, bool cnt) const
{
    std::ofstream myfile;
    myfile.open(filename);

    if (!myfile.is_open())
        throw "unable to open " + filename;

    const std::vector<label_counter> mappings = calibrate(dataSet);

    myfile << "CLASSES = {\n";
    for (size_t k=0; k < mappings.size(); ++k)
    {
        size_t y = k / m_width, x = k % m_width;
        myfile << " (" << y << ',' << x << "): { ";
        for (auto it = mappings[k].begin(); it != mappings[k].end(); ++it)
        {
            myfile << it->first;
            if (cnt)
            {
                myfile << ':' << it->second;
            }
            myfile << ", ";
        }
        myfile << "},\n";
    }
    myfile << "}\n";

    myfile.close();    
}



class Som : public _SomBase<double>
{
    using _SomBase<double>::_SomBase;

public:
    void train(const CSR&, size_t tmax, double, double, 
                double rN=FLT_MIN, double aN=FLT_MIN,
                double std=0.3, cooling rc=LINEAR, cooling ac=LINEAR);

private:
    void update(const CSR&, size_t, size_t, double, double, double);
    size_t getBmu(const CSR&, size_t, double&) const;
    void stabilize(size_t);

    double* squared_sum;
    double* w_coeff;
    double * wvprod;
};

class BSom : public _SomBase<float>
{
    using _SomBase<float>::_SomBase;

public:
    void train(const CSR&, size_t, float, float rN=0.f, 
                float std=0.3, cooling rc=LINEAR);

private:
    void updateEpoch(const CSR&, float, float, size_t *);
    void trainOneEpoch(const CSR&, float, float, size_t *, float *);
};

}

#endif // SOM_H_INCLUDED
