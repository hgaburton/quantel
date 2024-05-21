#ifndef EXCITATION_H
#define EXCITATION_H

#include <cstddef>

struct Excitation {
    size_t particle; // Particle index
    size_t hole; // Hole index
    bool spin; // Spin of the excitation

    inline bool operator< (const Excitation &rhs) const {
        if(particle < rhs.particle) return true;
        if(particle > rhs.particle) return false;
        if(hole < rhs.hole) return true;
        if(hole > rhs.hole) return false;
        return spin < rhs.spin;
    }
};

#endif // EXCITATION_H