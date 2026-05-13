#include "lookup_table.h"
#include <cmath>

namespace LookupTables {

namespace {
    const double t = std::sqrt(0.5);

    double A(double b, double x, double y) { return std::sqrt((b+x)/(b+y)); }
    double C(double b, double x)           { return std::sqrt((b+x-1.0)*(b+x+1.0))/(b+x); }
    double B(double b, double p, double q) { return std::sqrt(2.0/((b+p)*(b+q))); }
    double D(double b, double p)           { return std::sqrt(((b+p-1.0)*(2.0*p+2.0))/((b+p)*(b+p+1.0))); }
}

OBTableOne   ob_table_one;
OBTableTwo   ob_table_two;
TBTableOne   tb_table_one;
TBTableTwo   tb_table_two;
TBTableThree tb_table_three;
TBTableFour  tb_table_four;

void init_tables() {
    auto zero = [](double) { return 0.0; };

    for (auto& a : ob_table_one)   for (auto& b : a) for (auto& c : b) c = zero;
    for (auto& a : ob_table_two)   for (auto& b : a) for (auto& c : b) for (auto& d : c) d = zero;
    for (auto& a : tb_table_one)   for (auto& b : a) for (auto& c : b) for (auto& d : c) d = zero;
    for (auto& a : tb_table_two)   for (auto& b : a) for (auto& c : b) for (auto& d : c) d = zero;
    for (auto& a : tb_table_three) for (auto& b : a) for (auto& c : b) for (auto& d : c) for (auto& e : d) e = zero;
    for (auto& a : tb_table_four)  for (auto& b : a) for (auto& c : b) for (auto& d : c) for (auto& e : d) e = zero;

    // ==================== ob_table_one ====================
    // W
    ob_table_one[1][1][0] = [](double)   { return 1.0; };
    ob_table_one[2][2][0] = [](double)   { return 1.0; };
    ob_table_one[3][3][0] = [](double)   { return 2.0; };
    // hR
    ob_table_one[0][1][1] = [](double)   { return 1.0; };
    ob_table_one[0][2][1] = [](double)   { return 1.0; };
    ob_table_one[1][3][1] = [](double b) { return A(b, 0, 1); };
    ob_table_one[2][3][1] = [](double b) { return A(b, 2, 1); };
    // hL
    ob_table_one[1][0][2] = [](double)   { return 1.0; };
    ob_table_one[2][0][2] = [](double)   { return 1.0; };
    ob_table_one[3][1][2] = [](double b) { return A(b, 0, 1); };
    ob_table_one[3][2][2] = [](double b) { return A(b, 2, 1); };
    // tR
    ob_table_one[1][0][3] = [](double)   { return 1.0; };
    ob_table_one[2][0][3] = [](double)   { return 1.0; };
    ob_table_one[3][1][3] = [](double b) { return A(b, 1, 0); };
    ob_table_one[3][2][3] = [](double b) { return A(b, 1, 2); };
    // tL
    ob_table_one[0][1][4] = [](double)   { return 1.0; };
    ob_table_one[0][2][4] = [](double)   { return 1.0; };
    ob_table_one[1][3][4] = [](double b) { return A(b, 2, 1); };
    ob_table_one[2][3][4] = [](double b) { return A(b, 0, 1); };

    // ==================== ob_table_two ====================
    // R, deltab=-1
    ob_table_two[0][0][0][0] = [](double)   { return 1.0; };
    ob_table_two[1][1][0][0] = [](double)   { return -1.0; };
    ob_table_two[1][2][0][0] = [](double b) { return -1.0/(b+2.0); };
    ob_table_two[2][2][0][0] = [](double b) { return C(b, 2); };
    ob_table_two[3][3][0][0] = [](double)   { return -1.0; };
    // R, deltab=+1
    ob_table_two[0][0][0][1] = [](double)   { return 1.0; };
    ob_table_two[1][1][0][1] = [](double b) { return C(b, 0); };
    ob_table_two[2][1][0][1] = [](double b) { return 1.0/b; };
    ob_table_two[2][2][0][1] = [](double)   { return -1.0; };
    ob_table_two[3][3][0][1] = [](double)   { return -1.0; };
    // L, deltab=-1
    ob_table_two[0][0][1][0] = [](double)   { return 1.0; };
    ob_table_two[1][1][1][0] = [](double b) { return C(b, 1); };
    ob_table_two[1][2][1][0] = [](double b) { return 1.0/(b+1.0); };
    ob_table_two[2][2][1][0] = [](double)   { return -1.0; };
    ob_table_two[3][3][1][0] = [](double)   { return -1.0; };
    // L, deltab=+1
    ob_table_two[0][0][1][1] = [](double)   { return 1.0; };
    ob_table_two[1][1][1][1] = [](double)   { return -1.0; };
    ob_table_two[2][1][1][1] = [](double b) { return -1.0/(b+1.0); };
    ob_table_two[2][2][1][1] = [](double b) { return C(b, 1); };
    ob_table_two[3][3][1][1] = [](double)   { return -1.0; };

    // ==================== tb_table_one ====================
    // hRhR (=tLtL)
    tb_table_one[0][3][0][0] = [](double) { return std::sqrt(2.0); };
    // hLhL (=tRtR)
    tb_table_one[3][0][1][0] = [](double) { return std::sqrt(2.0); };
    // hRhL, x=0
    tb_table_one[1][1][2][0] = [](double)   { return t; };
    tb_table_one[2][2][2][0] = [](double)   { return t; };
    tb_table_one[3][3][2][0] = [](double)   { return 2.0*t; };
    // hRhL, x=1
    tb_table_one[1][1][2][1] = [](double b) { return -t*A(b, -1, 1); };
    tb_table_one[1][2][2][1] = [](double)   { return 1.0; };
    tb_table_one[2][1][2][1] = [](double)   { return 1.0; };
    tb_table_one[2][2][2][1] = [](double b) { return t*A(b, 3, 1); };
    // tRtL, x=0
    tb_table_one[1][1][3][0] = [](double)   { return t; };
    tb_table_one[2][2][3][0] = [](double)   { return -t; };
    tb_table_one[3][3][3][0] = [](double)   { return -2.0*t; };
    // tRtL, x=1
    tb_table_one[1][1][3][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_one[1][2][3][1] = [](double b) { return C(b, 2); };
    tb_table_one[2][1][3][1] = [](double b) { return C(b, 0); };
    tb_table_one[2][2][3][1] = [](double b) { return -t*A(b, 0, 2); };

    // ==================== tb_table_two ====================
    // hRtR (=tRhR)
    tb_table_two[1][1][0][0] = [](double)   { return 1.0; };
    tb_table_two[1][2][0][0] = [](double)   { return 1.0; };
    tb_table_two[2][1][0][1] = [](double)   { return 1.0; };
    tb_table_two[2][2][0][1] = [](double)   { return 1.0; };
    tb_table_two[3][3][0][0] = [](double)   { return 1.0; };
    tb_table_two[3][3][0][1] = [](double)   { return 1.0; };
    // hLtL (=tLhL)
    tb_table_two[1][1][1][1] = [](double)   { return 1.0; };
    tb_table_two[1][2][1][0] = [](double)   { return 1.0; };
    tb_table_two[2][1][1][1] = [](double)   { return 1.0; };
    tb_table_two[2][2][1][0] = [](double)   { return 1.0; };
    tb_table_two[3][3][1][0] = [](double)   { return 1.0; };
    tb_table_two[3][3][1][1] = [](double)   { return 1.0; };
    // hRtL
    tb_table_two[0][3][2][0] = [](double b) { return A(b, 2, 1); };
    tb_table_two[0][3][2][1] = [](double b) { return A(b, 0, 1); };
    // tRhL
    tb_table_two[0][3][3][0] = [](double b) { return A(b, 1, 1); };
    tb_table_two[0][3][3][1] = [](double b) { return A(b, 1, 0); };

    // ==================== tb_table_three ====================
    // RhR, deltab=-1, x=0
    tb_table_three[0][2][0][0][0] = [](double b) { return t*A(b, 1, 2); };
    tb_table_three[1][3][0][0][0] = [](double)   { return -t; };
    // RhR, deltab=-1, x=1
    tb_table_three[0][1][0][0][1] = [](double)   { return 1.0; };
    tb_table_three[0][2][0][0][1] = [](double b) { return t*A(b, 3, 2); };
    tb_table_three[1][3][0][0][1] = [](double b) { return -t*A(b, 0, 2); };
    tb_table_three[2][3][0][0][1] = [](double b) { return A(b, 3, 2); };
    // hRR, deltab=-1, x=0
    tb_table_three[0][2][1][0][0] = [](double b) { return t*A(b, 1, 2); };
    tb_table_three[1][3][1][0][0] = [](double)   { return -t; };
    // hRR, deltab=-1, x=1
    tb_table_three[0][1][1][0][1] = [](double)   { return -1.0; };
    tb_table_three[0][2][1][0][1] = [](double b) { return -t*A(b, 3, 2); };
    tb_table_three[1][3][1][0][1] = [](double b) { return t*A(b, 0, 2); };
    tb_table_three[2][3][1][0][1] = [](double b) { return -A(b, 3, 2); };
    // RhR, deltab=+1, x=0
    tb_table_three[0][1][0][1][0] = [](double b) { return t*A(b, 1, 0); };
    tb_table_three[2][3][0][1][0] = [](double)   { return -t; };
    // RhR, deltab=+1, x=1
    tb_table_three[0][1][0][1][1] = [](double b) { return -t*A(b, -1, 0); };
    tb_table_three[0][2][0][1][1] = [](double)   { return 1.0; };
    tb_table_three[1][3][0][1][1] = [](double b) { return A(b, -1, 0); };
    tb_table_three[2][3][0][1][1] = [](double b) { return t*A(b, 2, 0); };
    // hRR, deltab=+1, x=0
    tb_table_three[0][1][1][1][0] = [](double b) { return t*A(b, 1, 0); };
    tb_table_three[2][3][1][1][0] = [](double)   { return -t; };
    // hRR, deltab=+1, x=1
    tb_table_three[0][1][1][1][1] = [](double b) { return t*A(b, -1, 0); };
    tb_table_three[0][2][1][1][1] = [](double)   { return -1.0; };
    tb_table_three[1][3][1][1][1] = [](double b) { return -A(b, -1, 0); };
    tb_table_three[2][3][1][1][1] = [](double b) { return -t*A(b, 2, 0); };
    // hLL, deltab=-1, x=0
    tb_table_three[1][0][2][0][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_three[3][2][2][0][0] = [](double)   { return -t; };
    // LhL, deltab=-1, x=0
    tb_table_three[1][0][3][0][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_three[3][2][3][0][0] = [](double)   { return -t; };
    // hLL, deltab=-1, x=1
    tb_table_three[1][0][2][0][1] = [](double b) { return -t*A(b, 0, 1); };
    tb_table_three[2][0][2][0][1] = [](double)   { return 1.0; };
    tb_table_three[3][1][2][0][1] = [](double b) { return A(b, 0, 1); };
    tb_table_three[3][2][2][0][1] = [](double b) { return t*A(b, 3, 1); };
    // LhL, deltab=-1, x=1
    tb_table_three[1][0][3][0][1] = [](double b) { return t*A(b, 0, 1); };
    tb_table_three[2][0][3][0][1] = [](double)   { return -1.0; };
    tb_table_three[3][1][3][0][1] = [](double b) { return -A(b, 0, 1); };
    tb_table_three[3][2][3][0][1] = [](double b) { return -t*A(b, 3, 1); };
    // hLL, deltab=+1, x=0
    tb_table_three[2][0][2][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_three[3][1][2][1][0] = [](double)   { return -t; };
    // LhL, deltab=+1, x=0
    tb_table_three[2][0][3][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_three[3][1][3][1][0] = [](double)   { return -t; };
    // hLL, deltab=+1, x=1
    tb_table_three[1][0][2][1][1] = [](double)   { return 1.0; };
    tb_table_three[2][0][2][1][1] = [](double b) { return t*A(b, 2, 1); };
    tb_table_three[3][1][2][1][1] = [](double b) { return -t*A(b, -1, 1); };
    tb_table_three[3][2][2][1][1] = [](double b) { return A(b, 2, 1); };
    // LhL, deltab=+1, x=1
    tb_table_three[1][0][3][1][1] = [](double)   { return -1.0; };
    tb_table_three[2][0][3][1][1] = [](double b) { return -t*A(b, 2, 1); };
    tb_table_three[3][1][3][1][1] = [](double b) { return t*A(b, -1, 1); };
    tb_table_three[3][2][3][1][1] = [](double b) { return -A(b, 2, 1); };
    // hRL, deltab=-1: x=0 entries were overwritten by x=1 in the source, both land at [...,4,0,0]
    tb_table_three[0][1][4][0][0] = [](double)   { return 1.0; };
    tb_table_three[0][2][4][0][0] = [](double b) { return t*A(b, 3, 1); };
    tb_table_three[1][3][4][0][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_three[2][3][4][0][0] = [](double b) { return -A(b, 2, 1); };
    // hRL, deltab=+1, x=0
    tb_table_three[0][1][4][1][0] = [](double)   { return t; };
    tb_table_three[2][3][4][1][0] = [](double b) { return t*A(b, 0, 1); };
    // hRL, deltab=+1, x=1
    tb_table_three[0][1][4][1][1] = [](double b) { return -t*A(b, -1, 1); };
    tb_table_three[0][2][4][1][1] = [](double)   { return 1.0; };
    tb_table_three[1][3][4][1][1] = [](double b) { return -A(b, 0, 1); };
    tb_table_three[2][3][4][1][1] = [](double b) { return -t*A(b, 2, 1); };
    // RhL, deltab=-1, x=0
    tb_table_three[1][0][5][0][0] = [](double)   { return t; };
    tb_table_three[3][2][5][0][0] = [](double b) { return t*A(b, 1, 2); };
    // RhL, deltab=-1, x=1
    tb_table_three[1][0][5][0][1] = [](double b) { return -t*A(b, 0, 2); };
    tb_table_three[2][0][5][0][1] = [](double)   { return 1.0; };
    tb_table_three[3][1][5][0][1] = [](double b) { return -A(b, 1, 2); };
    tb_table_three[3][2][5][0][1] = [](double b) { return -t*A(b, 3, 2); };
    // RhL, deltab=+1, x=0
    tb_table_three[2][0][5][1][0] = [](double)   { return t; };
    tb_table_three[3][1][5][1][0] = [](double b) { return t*A(b, 1, 0); };
    // RhL, deltab=+1, x=1
    tb_table_three[1][0][5][1][1] = [](double)   { return 1.0; };
    tb_table_three[2][0][5][1][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_three[3][1][5][1][1] = [](double b) { return t*A(b, -1, 0); };
    tb_table_three[3][2][5][1][1] = [](double b) { return -A(b, 1, 0); };

    // ==================== tb_table_four ====================
    // tRR, deltab=-2, x=1
    tb_table_four[1][0][0][0][1] = [](double)   { return 1.0; };
    tb_table_four[3][2][0][0][1] = [](double b) { return A(b, 1, 2); };
    // RtR, deltab=-2, x=1
    tb_table_four[1][0][1][0][1] = [](double)   { return 1.0; };
    tb_table_four[3][2][1][0][1] = [](double b) { return -A(b, 1, 2); };
    // tRR, deltab=0, x=0
    tb_table_four[0][1][0][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[0][2][0][1][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[1][3][0][1][0] = [](double)   { return -t; };
    tb_table_four[2][3][0][1][0] = [](double)   { return -t; };
    // RtR, deltab=0, x=0
    tb_table_four[1][0][1][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[2][0][1][1][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[3][1][1][1][0] = [](double)   { return -t; };
    tb_table_four[3][2][1][1][0] = [](double)   { return -t; };
    // tRR, deltab=0, x=1
    tb_table_four[1][0][0][1][1] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[2][0][0][1][1] = [](double b) { return -t*A(b, 0, 1); };
    tb_table_four[3][1][0][1][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_four[3][2][0][1][1] = [](double b) { return -t*A(b, 0, 2); };
    // RtR, deltab=0, x=1
    tb_table_four[1][0][1][1][1] = [](double b) { return -t*A(b, 2, 1); };
    tb_table_four[2][0][1][1][1] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[3][1][1][1][1] = [](double b) { return -t*A(b, 2, 0); };
    tb_table_four[3][2][1][1][1] = [](double b) { return t*A(b, 0, 2); };
    // tRR, deltab=+2, x=1
    tb_table_four[2][0][0][2][1] = [](double)   { return 1.0; };
    tb_table_four[3][1][0][2][1] = [](double b) { return A(b, 1, 0); };
    // RtR, deltab=+2, x=1
    tb_table_four[2][0][1][2][1] = [](double)   { return -1.0; };
    tb_table_four[3][1][1][2][1] = [](double b) { return -A(b, 1, 0); };
    // LtL, deltab=-2, x=1
    tb_table_four[0][2][2][0][1] = [](double)   { return 1.0; };
    tb_table_four[1][3][2][0][1] = [](double b) { return A(b, 3, 2); };
    // tLL, deltab=-2, x=1
    tb_table_four[0][2][3][0][1] = [](double)   { return -1.0; };
    tb_table_four[1][3][3][0][1] = [](double b) { return -A(b, 3, 2); };
    // LtL, deltab=0, x=0
    tb_table_four[0][1][2][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[0][2][2][1][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[1][3][2][1][0] = [](double)   { return -t; };
    tb_table_four[2][3][2][1][0] = [](double)   { return -t; };
    // tLL, deltab=0, x=0
    tb_table_four[0][1][3][1][0] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[0][2][3][1][0] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[1][3][3][1][0] = [](double)   { return -t; };
    tb_table_four[2][3][3][1][0] = [](double)   { return -t; };
    // LtL, deltab=0, x=1
    tb_table_four[0][1][2][1][1] = [](double b) { return t*A(b, 2, 1); };
    tb_table_four[0][2][2][1][1] = [](double b) { return -t*A(b, 0, 1); };
    tb_table_four[1][3][2][1][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_four[2][3][2][1][1] = [](double b) { return -t*A(b, 0, 2); };
    // tLL, deltab=0, x=1
    tb_table_four[0][1][3][1][1] = [](double b) { return -t*A(b, 2, 1); };
    tb_table_four[0][2][3][1][1] = [](double b) { return t*A(b, 0, 1); };
    tb_table_four[1][3][3][1][1] = [](double b) { return -t*A(b, 2, 0); };
    tb_table_four[2][3][3][1][1] = [](double b) { return t*A(b, 0, 2); };
    // LtL, deltab=+2, x=1
    tb_table_four[0][1][2][2][1] = [](double)   { return 1.0; };
    tb_table_four[2][3][2][2][1] = [](double b) { return A(b, -1, 0); };
    // tLL, deltab=+2, x=1
    tb_table_four[0][1][3][2][1] = [](double)   { return -1.0; };
    tb_table_four[2][3][3][2][1] = [](double b) { return -A(b, -1, 0); };
    // RR, deltab=-2, x=1
    tb_table_four[0][0][4][0][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][4][0][1] = [](double)   { return 1.0; };
    tb_table_four[1][2][4][0][1] = [](double b) { return B(b, 2, 3); };
    tb_table_four[2][2][4][0][1] = [](double b) { return D(b, 2); };
    tb_table_four[3][3][4][0][1] = [](double)   { return 1.0; };
    // RR, deltab=0, x=0
    tb_table_four[0][0][4][1][0] = [](double)   { return 1.0; };
    tb_table_four[1][1][4][1][0] = [](double)   { return -1.0; };
    tb_table_four[2][2][4][1][0] = [](double)   { return -1.0; };
    tb_table_four[3][3][4][1][0] = [](double)   { return 1.0; };
    // RR, deltab=0, x=1
    tb_table_four[0][0][4][1][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][4][1][1] = [](double b) { return -D(b, 0); };
    tb_table_four[1][2][4][1][1] = [](double b) { return B(b, 1, 2); };
    tb_table_four[2][1][4][1][1] = [](double b) { return B(b, 0, 1); };
    tb_table_four[2][2][4][1][1] = [](double b) { return -D(b, 1); };
    tb_table_four[3][3][4][1][1] = [](double)   { return 1.0; };
    // RR, deltab=+2, x=1
    tb_table_four[0][0][4][2][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][4][2][1] = [](double b) { return D(b, -1); };
    tb_table_four[2][1][4][2][1] = [](double b) { return B(b, -1, 0); };
    tb_table_four[2][2][4][2][1] = [](double)   { return 1.0; };
    tb_table_four[3][3][4][2][1] = [](double)   { return 1.0; };
    // LL, deltab=-2, x=1
    tb_table_four[0][0][5][0][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][5][0][1] = [](double b) { return D(b, 1); };
    tb_table_four[1][2][5][0][1] = [](double b) { return B(b, 1, 2); };
    tb_table_four[2][2][5][0][1] = [](double)   { return 1.0; };
    tb_table_four[3][3][5][0][1] = [](double)   { return 1.0; };
    // LL, deltab=0, x=0
    tb_table_four[0][0][5][1][0] = [](double)   { return 1.0; };
    tb_table_four[1][1][5][1][0] = [](double)   { return -1.0; };
    tb_table_four[2][2][5][1][0] = [](double)   { return -1.0; };
    tb_table_four[3][3][5][1][0] = [](double)   { return 1.0; };
    // LL, deltab=0, x=1
    tb_table_four[0][0][5][1][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][5][1][1] = [](double b) { return -D(b, 0); };
    tb_table_four[1][2][5][1][1] = [](double b) { return B(b, 0, 1); };
    tb_table_four[2][1][5][1][1] = [](double b) { return B(b, 1, 2); };
    tb_table_four[2][2][5][1][1] = [](double b) { return -D(b, 1); };
    tb_table_four[3][3][5][1][1] = [](double)   { return 1.0; };
    // LL, deltab=+2, x=1
    tb_table_four[0][0][5][2][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][5][2][1] = [](double)   { return 1.0; };
    tb_table_four[2][1][5][2][1] = [](double b) { return B(b, 0, 1); };
    tb_table_four[2][2][5][2][1] = [](double b) { return D(b, 0); };
    tb_table_four[3][3][5][2][1] = [](double)   { return 1.0; };
    // RtL, deltab=-2, x=1
    tb_table_four[0][2][6][0][1] = [](double b) { return C(b, 2); };
    tb_table_four[1][3][6][0][1] = [](double b) { return -A(b, 3, 2); };
    // RtL, deltab=0, x=0
    tb_table_four[0][1][6][1][0] = [](double)   { return -t; };
    tb_table_four[0][2][6][1][0] = [](double)   { return -t; };
    tb_table_four[1][3][6][1][0] = [](double b) { return -t*A(b, 0, 1); };
    tb_table_four[2][3][6][1][0] = [](double b) { return -t*A(b, 2, 1); };
    // RtL, deltab=0, x=1
    tb_table_four[0][1][6][1][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_four[0][2][6][1][1] = [](double b) { return -t*A(b, 0, 2); };
    tb_table_four[1][3][6][1][1] = [](double b) { return -t*A(b, 2, 1); };
    tb_table_four[2][3][6][1][1] = [](double b) { return t*A(b, 0, 1); };
    // RtL, deltab=+2, x=1
    tb_table_four[0][1][6][2][1] = [](double b) { return C(b, 0); };
    tb_table_four[2][3][6][2][1] = [](double b) { return -A(b, -1, 0); };
    // tRL, deltab=-2, x=1
    tb_table_four[1][0][7][0][1] = [](double b) { return C(b, 2); };
    tb_table_four[3][2][7][0][1] = [](double b) { return -A(b, 1, 2); };
    // tRL, deltab=0, x=0
    tb_table_four[1][0][7][1][0] = [](double)   { return -t; };
    tb_table_four[2][0][7][1][0] = [](double)   { return -t; };
    tb_table_four[3][1][7][1][0] = [](double b) { return -t*A(b, 0, 1); };
    tb_table_four[3][2][7][1][0] = [](double b) { return -t*A(b, 2, 1); };
    // tRL, deltab=0, x=1
    tb_table_four[1][0][7][1][1] = [](double b) { return t*A(b, 2, 0); };
    tb_table_four[2][0][7][1][1] = [](double b) { return -t*A(b, 0, 2); };
    tb_table_four[3][1][7][1][1] = [](double b) { return -t*A(b, 2, 1); };
    tb_table_four[3][2][7][1][1] = [](double b) { return t*A(b, 0, 1); };
    // tRL, deltab=+2, x=1
    tb_table_four[2][0][7][2][1] = [](double b) { return C(b, 0); };
    tb_table_four[3][1][7][2][1] = [](double b) { return -A(b, 1, 0); };
    // RL, deltab=-2, x=1
    tb_table_four[0][0][8][0][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][8][0][1] = [](double b) { return -C(b, 2); };
    tb_table_four[1][2][8][0][1] = [](double b) { return -std::sqrt(2.0)/(b+2.0); };
    tb_table_four[2][2][8][0][1] = [](double b) { return -C(b, 2); };
    tb_table_four[3][3][8][0][1] = [](double)   { return 1.0; };
    // RL, deltab=0, x=0
    tb_table_four[0][0][8][1][0] = [](double)   { return 1.0; };
    tb_table_four[1][1][8][1][0] = [](double)   { return 1.0; };
    tb_table_four[2][2][8][1][0] = [](double)   { return 1.0; };
    tb_table_four[3][3][8][1][0] = [](double)   { return 1.0; };
    // RL, deltab=0, x=1
    tb_table_four[0][0][8][1][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][8][1][1] = [](double b) { return D(b, 0); };
    tb_table_four[1][2][8][1][1] = [](double b) { return -B(b, 0, 2); };
    tb_table_four[2][1][8][1][1] = [](double b) { return -B(b, 0, 2); };
    tb_table_four[2][2][8][1][1] = [](double b) { return D(b, 1); };
    tb_table_four[3][3][8][1][1] = [](double)   { return 1.0; };
    // RL, deltab=+2, x=1
    tb_table_four[0][0][8][2][1] = [](double)   { return 1.0; };
    tb_table_four[1][1][8][2][1] = [](double b) { return -C(b, 0); };
    tb_table_four[2][1][8][2][1] = [](double b) { return -std::sqrt(2.0)/b; };
    tb_table_four[2][2][8][2][1] = [](double b) { return -C(b, 0); };
    tb_table_four[3][3][8][2][1] = [](double)   { return 1.0; };
}

// Initialize immediately when the library loads
static bool initialized = []() {
    init_tables();
    return true;
}();


} // namespace LookupTables
