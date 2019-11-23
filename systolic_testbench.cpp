#include "systolic.h"
#include <stdlib.h>
using namespace std;
const int A_R = 342;
const int A_C = 271;
const int B_C = 140;
dtype tb_A[A_R * A_C];
dtype tb_B[A_C * B_C];
dtype tb_C[A_R * B_C];
int main()
{
    bool pass = true;
    for (int k = 0; k < 1; k++)
    {
        for (int i = 0; i < A_R; i++)
        {
            for(int j = 0; j< A_C; j++)
            {
                tb_A[A_C * i + j] = rand();
            }
        }
        for (int i = 0; i < A_C; i++)
        {
            for(int j = 0; j< B_C; j++)
            {
                tb_B[B_C * i + j] = rand();
            }
        }
        systolic(tb_A, tb_B, tb_C, A_R, A_C, B_C);
        for (int i = 0; i < A_R; i++)
        {
            for (int j = 0; j < B_C; j++)
            {
                dtype x = 0;
                for (int l = 0; l < A_C; l ++)
                {
                    x += tb_A[A_C * i + l] * tb_B[B_C * l + j];
                }
                if(x != tb_C[B_C * i + j])
                {
                    cerr<<i<<" "<<j<<" "<<x<<" "<<tb_C[B_C * i + j]<<endl;
                    pass = false;
                }
            }
        }
    }
    if(pass)
        return 0;
    else
        return 1;
}