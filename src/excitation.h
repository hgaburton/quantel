#ifndef EXCITATION_H
#define EXCITATION_H

#include <cstddef>

/// Single particle-hole excitation
struct Eph {
    size_t particle; // Particle index
    size_t hole; // Hole index

    inline bool operator< (const Eph &rhs) const {
        if(particle < rhs.particle) return true;
        if(particle > rhs.particle) return false;
        if(hole < rhs.hole) return true;
        return false;
    }
};

/// Double particle-hole excitation
struct Epphh {
    size_t particle1; // First particle index
    size_t particle2; // Second particle index
    size_t hole1; // First hole index
    size_t hole2; // Second hole index

    inline bool operator< (const Epphh &rhs) const {
        if(particle1 < rhs.particle1) return true;
        if(particle1 > rhs.particle1) return false;
        if(hole1 < rhs.hole1) return true;
        if(hole1 > rhs.hole1) return false;
        if(particle2 < rhs.particle2) return true;
        if(particle2 > rhs.particle2) return false;
        if(hole2 < rhs.hole2) return true;
        return false;
    }

    inline bool operator!= (const Epphh &rhs) const {
        return (particle1 != rhs.particle1) || (particle2 != rhs.particle2) ||
               (hole1 != rhs.hole1) || (hole2 != rhs.hole2);
    }
};

#endif // EXCITATION_H