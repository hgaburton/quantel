#ifndef PERIODIC_TABLE_H
#define PERIODIC_TABLE_H

#include <map>
#include <string>

const std::map<std::string,int> periodic_table{
    { "H", 1},                                                                                                                                                      {"HE", 2},
    {"LI", 3},{"BE", 4},                                                                                          { "B", 5},{ "C", 6},{ "N", 7},{ "O", 8},{ "F", 9},{"NE",10},
    {"NA",11},{"MG",12},                                                                                          {"AL",13},{"SI",14},{ "P",15},{ "S",16},{"CL",17},{"AR",18},
    { "K",19},{"CA",20},{"SC",21},{"TI",22},{ "V",23},{"MN",24},{"FE",25},{"CO",26},{"NI",27},{"CU",28},{"ZN",29},{"GA",30},{"GE",31},{"AS",32},{"SE",33},{"BR",34},{"KR",35}
};

const std::map<int,std::string> element_labels{
    { 1, "H"},                                                                                                                                                      { 2,"He"},
    { 3,"Li"},{ 4,"Be"},                                                                                          { 5, "B"},{ 6, "C"},{ 7, "N"},{ 8, "O"},{ 9, "F"},{10,"Ne"},
    {11,"Na"},{12,"Mg"},                                                                                          {13,"Al"},{14,"Si"},{15, "P"},{16, "S"},{17,"Cl"},{18,"Ar"},
    {19, "K"},{20,"Ca"},{21,"Sc"},{22,"Ti"},{23, "V"},{24,"Mn"},{25,"Fe"},{26,"Co"},{27,"Ni"},{28,"Cu"},{29,"Zn"},{30,"Ga"},{31,"Ge"},{32,"As"},{33,"Se"},{34,"Br"},{35,"Kr"}
};

#endif // PERIODIC_TABLE_H