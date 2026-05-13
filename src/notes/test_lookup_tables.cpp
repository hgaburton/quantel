#include "lookup_table.h"
#include <cstdio>

// Prints all table entries at three b values as CSV:
//   table_name,b_idx,i,j[,k[,l[,m]]],value
// b values: b_idx 0->2.0  1->3.0  2->5.0

int main() {
    LookupTables::init_tables();
    using namespace LookupTables;

    const double bs[3] = {2.0, 3.0, 5.0};

    for (int bi = 0; bi < 3; bi++) {
        const double b = bs[bi];

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<5;k++)
            std::printf("ob_table_one,%d,%d,%d,%d,%.15g\n", bi,i,j,k, ob_table_one[i][j][k](b));

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<2;k++) for (int l=0;l<2;l++)
            std::printf("ob_table_two,%d,%d,%d,%d,%d,%.15g\n", bi,i,j,k,l, ob_table_two[i][j][k][l](b));

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) for (int l=0;l<2;l++)
            std::printf("tb_table_one,%d,%d,%d,%d,%d,%.15g\n", bi,i,j,k,l, tb_table_one[i][j][k][l](b));

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<4;k++) for (int l=0;l<2;l++)
            std::printf("tb_table_two,%d,%d,%d,%d,%d,%.15g\n", bi,i,j,k,l, tb_table_two[i][j][k][l](b));

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<6;k++) for (int l=0;l<2;l++) for (int m=0;m<2;m++)
            std::printf("tb_table_three,%d,%d,%d,%d,%d,%d,%.15g\n", bi,i,j,k,l,m, tb_table_three[i][j][k][l][m](b));

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) for (int k=0;k<9;k++) for (int l=0;l<3;l++) for (int m=0;m<2;m++)
            std::printf("tb_table_four,%d,%d,%d,%d,%d,%d,%.15g\n", bi,i,j,k,l,m, tb_table_four[i][j][k][l][m](b));
    }
    return 0;
}
