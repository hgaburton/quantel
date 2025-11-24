#ifndef TWO_ARRAY_H
#define TWO_ARRAY_H

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept> 

/// \brief Class for handling two-index arrays
class TwoArray {
private:
    std::string m_sym; // Symmetry type: {'s1', 's2'}
protected:
    size_t m_dim[2];
    std::vector<double> m_data;

    /// \brief Get index into the two-index array
    size_t index(size_t p, size_t q) const 
    {
        return p*m_dim[1] + q;
    }

public:
    /// \brief Default constructor
    TwoArray() : m_dim{0,0}, m_data() { }
    
    /// \brief Copy constructor
    TwoArray(const TwoArray &other) :
        m_dim{other.m_dim[0], other.m_dim[1]}, m_data(other.m_data)
    { } 

    /// \brief Constructor
    TwoArray(std::vector<double> data, size_t dim1, size_t dim2) :
        m_dim{dim1,dim2}, m_data(data)
    {
        // Check dimensions 
        if(data.size() != dim1 * dim2) {
            throw std::invalid_argument("Data size does not match expected size for given dimensions.");
        }
    }

    /// \brief Default destructor
    virtual ~TwoArray() { }

    /// Access element (modifiable)
    double& operator()(size_t p, size_t q) {
        if(p>=m_dim[0]) throw std::out_of_range("TwoArray: Index 0 out of range");
        if(q>=m_dim[1]) throw std::out_of_range("TwoArray: Index 1 out of range");
        return m_data[index(p,q)];        
    }

    /// Access element (const)
    const double& operator()(size_t p, size_t q) const {
        if(p>=m_dim[0]) throw std::out_of_range("TwoArray: Index 0 out of range");
        if(q>=m_dim[1]) throw std::out_of_range("TwoArray: Index 1 out of range");
        return m_data[index(p,q)];
    }

    /// Get dimensions
    std::tuple<size_t, size_t> dim() const {
        return std::make_tuple(m_dim[0], m_dim[1]);
    }

    /// Resize and initialize to zero
    void resize(size_t dim1, size_t dim2) {
        m_dim[0] = dim1;
        m_dim[1] = dim2;
        m_data.resize(dim1*dim2);
        std::fill(m_data.begin(), m_data.end(), 0.0);
    }
};

#endif // TWO_ARRAY_H