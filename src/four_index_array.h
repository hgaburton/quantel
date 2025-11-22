#ifndef FOUR_INDEX_ARRAY_H
#define FOUR_INDEX_ARRAY_H

#include <vector>

/// \brief Class for handling four-index arrays
class FourIndexArray {
private:
        std::string m_sym; // Symmetry type: {'s1', 's4', 's8'}
protected:
    size_t m_dim[4];
    std::vector<double> m_data;

    /// \brief Get index into the four-index array
    size_t index(size_t p, size_t q, size_t r, size_t s) const 
    {
        if(m_sym=="s1") 
        { 
            // No symmetry
            return p*m_dim[1]*m_dim[2]*m_dim[3] + q*m_dim[2]*m_dim[3] + r*m_dim[3] + s;
        } 
        else if (m_sym=="s4") 
        {
            // <pq|rs> = <ps|rq> = <rq|ps> = <rs|pq>
            // This means we want p<r and q<s
            size_t p_ = std::min(p,r);
            size_t q_ = std::max(p,r);
            size_t r_ = std::min(q,s);
            size_t s_ = std::max(q,s);
            return p_*m_dim[1]*m_dim[2]*m_dim[3] + q_*m_dim[2]*m_dim[3] + r_*m_dim[3] + s_;
        } 
        else if (m_sym=="s8") 
        {
            // <pq|rs> = <qp|sr> = <rs|pq> = <sr|qp>
            // This means we want p<=q and r<=s, and (p,q)<=(r,s)
            bool pq_rs = (p*q < r*s);
            size_t p_ = (pq_rs) ? std::min(p,q) : std::min(r,s);
            size_t q_ = (pq_rs) ? std::max(p,q) : std::max(r,s);
            size_t r_ = (pq_rs) ? std::min(r,s) : std::min(p,q);
            size_t s_ = (pq_rs) ? std::max(r,s) : std::max(p,q);
            return p_*m_dim[1]*m_dim[2]*m_dim[3] + q_*m_dim[2]*m_dim[3] + r_*m_dim[3] + s_;
        }
    }

public:

    /// \brief Constructor
    FourIndexArray(std::vector<double> data, size_t dim1, size_t dim2, size_t dim3, size_t dim4, std::string sym="s1") :
        m_dim{dim1, dim2, dim3, dim4}, m_sym(sym), m_data(data)
    {
        // Check that data size matches dimensions
        size_t expected_size = 0;
        if(m_sym=="s1") {
            expected_size = m_dim[0] * m_dim[1] * m_dim[2] * m_dim[3];
        } else if (m_sym=="s4") {
            if(m_dim[0] != m_dim[2] || m_dim[1] != m_dim[3]) {
                throw std::invalid_argument("Dimensions must match for s4 symmetry.");
            }
            expected_size = (m_dim[0]*(m_dim[0]+1)/2) * (m_dim[1]*(m_dim[1]+1)/2);
        } else if (m_sym=="s8") {
            if(m_dim[0] != m_dim[1] || m_dim[0] != m_dim[2] || m_dim[0] != m_dim[3]) {
                throw std::invalid_argument("Dimensions must match for s8 symmetry.");
            }
            expected_size = (m_dim[0]*(m_dim[0]+1)/2)*(m_dim[0]*(m_dim[0]+1)/2)/2;
        }
        if(data.size() != expected_size) {
            throw std::invalid_argument("Data size does not match expected size for given dimensions and symmetry.");
        }
    }

    /// \brief Default destructor
    virtual ~FourIndexArray() { }

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
};

#endif // FOUR_INDEX_ARRAY_H