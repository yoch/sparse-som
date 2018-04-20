#include "som.h"



using namespace std;
using namespace som;


size_t NBSTABLES = 0;
#define EPSILON 1.e-50

/*
//#define HEX_H 0.8660254037844386
#define SQRT2 1.4142135623730951
*/

//////////////////////////////////////// Som ////////////////////////////////////////////////

static inline double euclideanDistanceSq(const float * const vsp, const int * const ind, const size_t vsz, 
                                         const double x2, const double * const w, const double w2, 
                                         const double wcoeff)
{
    return max(0., w2 - 2 * wcoeff * prod(w, vsp, ind, vsz) + x2);
}

static inline void partial_update(const float * const vsp, const int * const ind, const size_t n, 
                                  double * const w, const double vcoeff)
{
    for (size_t it=0; it<n; ++it)
    {
        w[ind[it]] += vsp[it] * vcoeff;
    }
}


void Som::train(const CSR& data, size_t tmax,
           double radius0, double alpha0, double radiusN, double alphaN,
           double stdCoeff, cooling rcool, cooling acool)
{
    NBSTABLES = 0;

    double tm = 0.;

    if (m_verbose > 0)
    {
        cout << "Start learning..." << endl;

        tm = get_wall_time();
    }

    // allocate helpers internal arrays
    squared_sum = new double[m_height * m_width];
    w_coeff = new double[m_height * m_width];
    wvprod = new double[m_height * m_width];

#pragma omp parallel for
    // Init w^2 before training
    for (idx_t k=0; k < m_height * m_width; ++k)
    {
        squared_sum[k] = squared(codebook + k * m_dim, m_dim);
    }

    // init coefficients
    fill_n(w_coeff, m_height * m_width, 1.);

    double Qe = 0;
    const size_t percent = tmax / 100;

    default_random_engine rng(time(NULL));
    uniform_int_distribution<size_t> uidist(0, data.nrows-1);

    for (size_t t=1; t <= tmax; ++t)
    {
        double radius, alpha;
        const double ratio = ((double)t / tmax);

        switch (rcool) {
        case EXPONENTIAL:
            radius = radius0 * pow(radiusN / radius0, ratio);
            break;
        case LINEAR:
        default:
            radius = radiusN + (radius0 - radiusN) * (1. - ratio);
            break;
        }

        switch (acool) {
        case EXPONENTIAL:
            alpha = alpha0 * pow(alphaN / alpha0, ratio);
            break;
        case LINEAR:
        default:
            alpha = alphaN + (alpha0 - alphaN) * (1. - ratio);
            break;
        }

        const size_t k = uidist(rng);

        // Get the best match unit
        double dStar;
        const size_t kStar = getBmu(data, k, dStar);

        // Update network
        update(data, k, kStar, radius, alpha, stdCoeff);

        if (m_verbose > 1)
        {
            Qe += sqrt(dStar);  // approximated

            if (t % percent == 0)
            {
                cout << "  " << t / percent << "% (" << t << " / " << tmax << ")"
                    << " - r = " << radius << ", a = " << alpha
                    << " - approx. QE " << Qe / percent << endl;
                Qe = 0; // reinit Qe count
            }
        }
    }

    // stabilize all units
#pragma omp parallel for \
    schedule(static,1)

    for (idx_t k=0; k < m_height * m_width; ++k)
    {
        stabilize(k);
    }

    // dealloc internal helpers arrays
    delete [] squared_sum;
    delete [] w_coeff;
    delete [] wvprod;

    if (m_verbose > 0)
    {
        tm = get_wall_time() - tm;

        cout << "Finished: elapsed : " << tm << "s" << endl;
        if (m_verbose > 1)
        {
            cout << "  coeffs rescaled " << NBSTABLES << " times" << endl;
        }

    }
}

void Som::update(const CSR& data, size_t n, size_t kStar, double radius, double alpha, double stdCoef)
{
    const int x = kStar % m_width;
    const int y = kStar / m_width;
    const double m2rcfinv = -1. / (2. * squared(radius * stdCoef));
/*
    const int r = radius; // +1 ?
    const int startI = y-r,
              stopI = y+r,
              startJ = x-r,
              stopJ = x+r;
*/
    const int ind = data.indptr[n];
    const int vsz = data.indptr[n+1] - ind;
    const float * const vsp = &data.data[ind];
    const int * const vind = &data.indices[ind];

//#pragma omp parallel for collapse(2)
    //schedule(static,stopJ-startJ+1)
    //if ((stopI-startI)*(stopJ-startJ)>=100)

    for(int i = 0; i < (int)m_height; ++i)
    {
        for(int j = 0; j < (int)m_width; ++j)
        {
            double d2;

            if (fdist(y, x, i, j, d2) > radius+1)
                continue;

            const size_t idx = i * m_width + j;
            const double neighborhood = exp(d2 * m2rcfinv);
            const double a = alpha * neighborhood;
            const double b = 1. - a; // beware, if b==0 then calculus goes wrong

            // calculate and store {w_{t+1}}^2
            squared_sum[idx] = squared(b) * squared_sum[idx] +
                               2 * a * b * w_coeff[idx] * wvprod[idx] +
                               squared(a) * data._sqsum[n];

            w_coeff[idx] *= b;  // do that before using in expression below

            partial_update(vsp, vind, vsz, codebook + idx * m_dim, a / w_coeff[idx]);

            if (w_coeff[idx] < EPSILON)
            {
                stabilize(idx);
            }
        }
    }
}

size_t Som::getBmu(const CSR& data, const size_t n, double& dStar) const
{
    const size_t ind = data.indptr[n];
    const size_t vsz = data.indptr[n+1] - ind;
    const float * const vsp = &data.data[ind];
    const int * const vind = &data.indices[ind];

    bmu mini = bmu(-1, DBL_MAX);
/*
#pragma omp declare \
    reduction(BMU : bmu : omp_out = omp_in.dst <= omp_out.dst ? omp_in : omp_out) \
    initializer (omp_priv(omp_orig))

#pragma omp parallel for \
    reduction(BMU:mini) \
    schedule(static,1)
*/
    for (size_t k=0; k < m_height*m_width; ++k)
    {
        //precompute w*v (to be used at update stage)
        wvprod[k] = prod(codebook + k * m_dim, vsp, vind, vsz);

        // fast squared euclidean dst (invariant by v^2) : w^2 - 2 w.v
        const double dst = squared_sum[k] - 2 * (wvprod[k] * w_coeff[k]);
        if (dst < mini.dst)
        {
            mini.idx = k;
            mini.dst = dst;
        }
    }

    // store values
    dStar = max(0., mini.dst + data._sqsum[n]); // correct euclidean dist
    return mini.idx;
}

void Som::stabilize(size_t k)
{
    const double coeff = w_coeff[k];
    double * const w = codebook + k * m_dim;
    for (size_t i=0; i < m_dim; ++i)
    {
        w[i] *= coeff;
    }
    w_coeff[k] = 1.;
#pragma omp atomic
    ++NBSTABLES;
}
