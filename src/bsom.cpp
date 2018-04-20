#include "som.h"
/*
#include <random>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>

#if defined(_OPENMP)
    #include <omp.h>
#endif
*/

using namespace std;
using namespace som;

/*
#if defined(_MSC_VER)
    using idx_t = int;
#else
    using idx_t = unsigned int;
#endif
*/

//////////////////////////////////////// BSom ////////////////////////////////////////////////

void BSom::train(const CSR& data, size_t tmax, float radius0, float radiusN, float stdCoef, cooling rcool)
{
    double tm = 0.;
    float radius;

    if (m_verbose > 0)
    {
        cout << "Start learning..." << endl;

        tm = get_wall_time();
    }

    size_t * bmus = new size_t[data.nrows];
    float * dsts = new float[data.nrows];

    for (size_t t=1; t<=tmax; ++t)
    {
        const float ratio = (double)(t-1) / (tmax-1);

        switch (rcool) {
        case EXPONENTIAL:
            radius = radius0 * powf(radiusN / radius0, ratio);
            break;
        case LINEAR:
        default:
            radius = radiusN + (radius0 - radiusN) * (1. - ratio);
            break;
        }

        trainOneEpoch(data, radius, stdCoef, bmus, dsts);

        if (m_verbose > 1)
        {
            cout << "  epoch " << t << " / " << tmax;
            cout << ", r = " << radius;
        }
    }

    if (m_verbose > 0)
    {

/* DBG */
        if (m_verbose > 1)
        {
            size_t * second = new size_t[data.nrows];
            float * sdsts = new float[data.nrows];
            
            getBmus(data, bmus, dsts, second, sdsts);

            // unable to compute QE if we don't have X^2
            if (data._sqsum)
            {
                double Qe = 0;
                for (int i=0; i<data.nrows; ++i)
                {
                    Qe += sqrt(max(0.f, dsts[i] + data._sqsum[i]));
                }
                Qe /= data.nrows;

                // Note: in fact, this is the quantization error of the previous step (before the update)
                cout << " - QE: " << Qe;
            }

            if (second)
            {
                // Note: in fact, this is the topographic error of the previous step (before the update)
                cout << " - TE: " << topographicError(bmus, second, data.nrows);
            }
            
            delete [] second;
            delete [] sdsts;

            cout << endl;
        }

        tm = get_wall_time() - tm;

        cout << "Finished: elapsed " << tm << "s" << endl;
    }
    
    delete [] bmus;
    delete [] dsts;
}

void BSom::updateEpoch(const CSR& data, const float radius, const float stdCoef, size_t * const bmus)
{
//    SKIPPED = 0;
    const double m2rcfinv = -1. / (2. * squared(radius * stdCoef));

#pragma omp parallel // shared(data)
 {
    // init an empty vector (tmp) by thread
    double * const numerator = new double[m_dim];

#pragma omp for

    for (idx_t xy=0; xy < m_height * m_width; ++xy)
    {
        const int y = xy / m_width,
                  x = xy % m_width;

        // init numerator and denominator
        double denominator = 0.f;
        fill_n(numerator, m_dim, 0.f);

        for (size_t n=0; n < (size_t) data.nrows; ++n)
        {
            const int ind = data.indptr[n];
            const int vsz = data.indptr[n+1] - ind;
            const float * const vsp = &data.data[ind];
            const int * const vind = &data.indices[ind];

            const int i = bmus[n] / m_width,
                      j = bmus[n] % m_width;

            double d2;

            if (fdist(y, x, i, j, d2) > radius+1)
                continue;

            const double h = exp(d2 * m2rcfinv);

            denominator += h;
            for (int it=0; it<vsz; ++it)
            {
                numerator[vind[it]] += h * vsp[it];
            }
        }

        if (denominator != 0)
        {
            float * const w = codebook + xy * m_dim;
            for (size_t k=0; k < m_dim; ++k)
            {
                w[k] = numerator[k] / denominator;
            }
        }
/*
        else
        {
#pragma omp atomic
            SKIPPED++;
        }
*/
    }
    delete [] numerator;
 }
}

void BSom::trainOneEpoch(const CSR& data, float radius, float stdCoef, 
                        size_t * const bmus, float * const dsts)
{

    size_t * second = NULL;
    float * sdsts = NULL;

    if (m_verbose > 1)
    {
        second = new size_t[data.nrows];
        sdsts = new float[data.nrows];
    }

    ///////////////////////////////////////////
    ///          Searching BMUs
    getBmus(data, bmus, dsts, second, sdsts);

    //////////////////////////////////////////
    ///            Update phase
    updateEpoch(data, radius, stdCoef, bmus);

    if (m_verbose > 1)
    {

        // unable to compute QE if we don't have X^2
        if (data._sqsum)
        {
            double Qe = 0;
            for (int i=0; i<data.nrows; ++i)
            {
                Qe += sqrt(max(0.f, dsts[i] + data._sqsum[i]));
            }
            Qe /= data.nrows;

            // Note: in fact, this is the quantization error of the previous step (before the update)
            cout << " - QE: " << Qe;
        }

        if (second)
        {
            cout << " - TE: " << topographicError(bmus, second, data.nrows);
        }

//        cout << "  (" << SKIPPED << " units skipped)";
        // Note: in fact, this is the topographic error of the previous step (before the update)
        cout << endl;

    }

    if (second) delete [] second;
    if (sdsts) delete [] sdsts;
}
