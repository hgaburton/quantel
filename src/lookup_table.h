#ifndef LOOKUP_TABLE_H
#define LOOKUP_TABLE_H

#include <array>
#include <functional>

namespace LookupTables {

using Func = std::function<double(double)>;

// ob_table_one[d'][d][op]
//   op: W=0, hR=1, hL=2, tR=3, tL=4
using OBTableOne   = std::array<std::array<std::array<Func, 5>, 4>, 4>;

// ob_table_two[d'][d][RL][deltab]
//   RL: R=0, L=1 | deltab: -1->0, +1->1
using OBTableTwo   = std::array<std::array<std::array<std::array<Func, 2>, 2>, 4>, 4>;

// tb_table_one[d'][d][op][x]
//   op: hRhR(=tLtL)=0, hLhL(=tRtR)=1, hRhL=2, tRtL=3
using TBTableOne   = std::array<std::array<std::array<std::array<Func, 2>, 4>, 4>, 4>;

// tb_table_two[d'][d][op][deltab]
//   op: hRtR(=tRhR)=0, hLtL(=tLhL)=1, hRtL=2, tRhL=3
//   deltab: -1->0, +1->1
using TBTableTwo   = std::array<std::array<std::array<std::array<Func, 2>, 4>, 4>, 4>;

// tb_table_three[d'][d][op][deltab][x]
//   op: RhR=0, hRR=1, hLL=2, LhL=3, hRL=4, RhL=5
//   deltab: -1->0, +1->1
using TBTableThree = std::array<std::array<std::array<std::array<std::array<Func, 2>, 2>, 6>, 4>, 4>;

// tb_table_four[d'][d][op][deltab][x]
//   op: tRR=0, RtR=1, LtL=2, tLL=3, RR=4, LL=5, RtL=6, tRL=7, RL=8
//   deltab: -2->0, 0->1, +2->2
using TBTableFour  = std::array<std::array<std::array<std::array<std::array<Func, 2>, 3>, 9>, 4>, 4>;

extern OBTableOne   ob_table_one;
extern OBTableTwo   ob_table_two;
extern TBTableOne   tb_table_one;
extern TBTableTwo   tb_table_two;
extern TBTableThree tb_table_three;
extern TBTableFour  tb_table_four;

// Must be called once before accessing any table.
void init_tables();

} // namespace LookupTables

#endif // LOOKUP_TABLE_H
