#ifndef FOUR_ARRAY_H
#define FOUR_ARRAY_H

#include <vector>
#include <string>        // added
#include <algorithm>     // for std::min/std::max
#include <stdexcept> 

/// \brief Class for handling four-index arrays
class FourArray {

protected:
    size_t m_dim[4];
    std::vector<double> m_data;

    /// \brief Get index into the four-index array
    size_t index(size_t p, size_t q, size_t r, size_t s) const 
    {
        return p*m_dim[1]*m_dim[2]*m_dim[3] + q*m_dim[2]*m_dim[3] + r*m_dim[3] + s;
    }

public:
    /// \brief Default constructor
    FourArray() : m_dim{0,0,0,0}, m_data()
    { }

    /// \brief Copy constructor
    FourArray(const FourArray &other) :
        m_dim{other.m_dim[0], other.m_dim[1], other.m_dim[2], other.m_dim[3]}, m_data(other.m_data)
    { }

    /// \brief Constructor
    FourArray(std::vector<double> data, size_t dim1, size_t dim2, size_t dim3, size_t dim4) :
        m_dim{dim1, dim2, dim3, dim4}, m_data(data)
    {
        // Check dimensions
        if(data.size() != dim1 * dim2 * dim3 * dim4) {
            throw std::invalid_argument("Data size does not match expected size for given dimensions.");
        }
    }

    /// \brief Default destructor
    virtual ~FourArray() { }

    /// Access element (modifiable)
    double& operator()(size_t p, size_t q, size_t r, size_t s) {
        if(p>=m_dim[0]) throw std::out_of_range("Index 0 out of range");
        if(q>=m_dim[1]) throw std::out_of_range("Index 1 out of range");
        if(r>=m_dim[2]) throw std::out_of_range("Index 2 out of range");
        if(s>=m_dim[3]) throw std::out_of_range("Index 3 out of range");
        return m_data[index(p,q,r,s)];        
    }

    /// Access element (const)
    const double& operator()(size_t p, size_t q, size_t r, size_t s) const {
        if(p>=m_dim[0]) throw std::out_of_range("Index 0 out of range");
        if(q>=m_dim[1]) throw std::out_of_range("Index 1 out of range");
        if(r>=m_dim[2]) throw std::out_of_range("Index 2 out of range");
        if(s>=m_dim[3]) throw std::out_of_range("Index 3 out of range");
        return m_data[index(p, q, r, s)];
    }

    /// Get dimensions
    std::tuple<size_t, size_t, size_t, size_t> dim() const {
        return std::make_tuple(m_dim[0], m_dim[1], m_dim[2], m_dim[3]);
    }
};

#endif // FOUR_ARRAY_H