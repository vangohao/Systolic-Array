#include "systolic.h"
#include <iostream>
#include "hls_stream.h"
#define stmtype hls::stream<dtype>
const int NN = 64;
const int N = 8;
dtype A0[NN][NN];
dtype B0[NN][NN];
dtype A1[NN][NN];
dtype B1[NN][NN];
dtype C[NN][NN];

void provide_A(dtype A[NN][NN], stmtype hpipe[N + 1][N + 1], int i, int ar)
{
    for (int k = 0; k<NN; k++)
    {
        hpipe[i][0].write(A[ar * N + i - 1][k]);
    }
}

void provide_B(dtype B[NN][NN], stmtype vpipe[N + 1][N + 1], int j, int bc)
{
    for (int k = 0; k<NN; k++)
    {
        vpipe[0][j].write(B[k][bc * N + j - 1]);
    }
}

void eat_h(stmtype hpipe[N + 1][N + 1], int i)
{
    dtype eat;
    for (int k = 0; k<NN; k++)
    {
        eat = hpipe[i][N].read();
    }
}

void eat_v(stmtype vpipe[N + 1][N + 1], int j)
{
    dtype eat;
    for (int k = 0; k<NN; k++)
    {
        eat = vpipe[N][j].read();
    }
}

void PE(stmtype hpipe[N + 1][N + 1], stmtype vpipe[N + 1][N + 1], int i, int j, int ar, int bc)
{
    #pragma HLS stream variable=hpipe dim=1 depth=3
    #pragma HLS stream variable=hpipe dim=2 depth=3
    #pragma HLS stream variable=vpipe dim=1 depth=3
    #pragma HLS stream variable=vpipe dim=2 depth=3

    dtype a;
    dtype b;
    for(int k = 0; k < NN; k++)
    {
    #pragma HLS pipeline
        // Shift_in(a, b, A, B, hpipe, vpipe, i, j, k);
        a = hpipe[i][j - 1].read();
        b = vpipe[i - 1][j].read();
        C[ar * N + i - 1][bc * N + j - 1] += a * b;
        hpipe[i][j].write(a);
        vpipe[i][j].write(b);
        // Shift_out(a, b, hpipe, vpipe, i, j);
    }
}

void PE_array(dtype A[NN][NN], dtype B[NN][NN], int ar, int bc)
{
    #pragma HLS array_partition variable=A cyclic factor=N dim=1
    #pragma HLS array_partition variable=B cyclic factor=N dim=2
    #pragma HLS array_partition variable=C cyclic factor=N dim=1
    #pragma HLS array_partition variable=C cyclic factor=N dim=2

    #pragma HLS dataflow
stmtype hpipe[N + 1][N + 1];  // Output Pipe
stmtype vpipe[N + 1][N + 1];
    #pragma HLS stream variable=hpipe dim=1 depth=3
    #pragma HLS stream variable=hpipe dim=2 depth=3
    #pragma HLS stream variable=vpipe dim=1 depth=3
    #pragma HLS stream variable=vpipe dim=2 depth=3
    
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        provide_A(A, hpipe, i, ar);
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        provide_B(B, vpipe, i, bc);
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        for (int j = 1; j<=N; j++)
        {
            #pragma HLS UNROLL
            PE(hpipe, vpipe, i, j, ar, bc);
        }
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        eat_v(vpipe, i);
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        eat_h(hpipe, i);
    }
}

void load_data(dtype A[NN][NN], dtype B[NN][NN], dtype *input_A, dtype *input_B, int offsetA, int offsetB, int A_R, int A_C, int B_C, int ars, int acs, int bcs)
{
    for (int i = 0; i < NN; i++)
    {
        for (int j = 0; j < NN; j++)
        {
            #pragma HLS pipeline
            if (i < ars && j < acs)
                A[i][j] = input_A[offsetA + i * A_C + j];
            else
                A[i][j] = 0;
        }
    }

    for (int i = 0; i < NN; i++)
    {
        for (int j = 0; j<NN; j++)
        {
            #pragma HLS pipeline
            if (i < acs && j < bcs)
                B[i][j] = input_B[offsetB + i * B_C + j];
            else
                B[i][j] = 0;
        }
    }
}

void systolic(dtype *input_A, dtype *input_B, dtype *output_C, int A_R, int A_C, int B_C)
{
    #pragma HLS INTERFACE s_axilite port = return
    #pragma HLS INTERFACE s_axilite port = A_R
    #pragma HLS INTERFACE s_axilite port = A_C
    #pragma HLS INTERFACE s_axilite port = B_C
    #pragma HLS INTERFACE m_axi depth = 256 port = input_A offset = slave
    #pragma HLS INTERFACE m_axi depth = 256 port = input_B offset = slave
    #pragma HLS INTERFACE m_axi depth = 256 port = output_C offset = slave
    #pragma HLS array_partition variable=A0 cyclic factor=N dim=1
    #pragma HLS array_partition variable=A1 cyclic factor=N dim=1
    #pragma HLS array_partition variable=B0 cyclic factor=N dim=2
    #pragma HLS array_partition variable=B1 cyclic factor=N dim=2
    #pragma HLS array_partition variable=C cyclic factor=N dim=1
    #pragma HLS array_partition variable=C cyclic factor=N dim=2

    for(int offset_A_R = 0; offset_A_R < A_R; offset_A_R += NN)
    {
        for(int offset_B_C = 0; offset_B_C < B_C; offset_B_C += NN)
        {
            // std::cerr<<offset_A_R<<","<<offset_B_C<<std::endl;
            for (int i = 0; i < NN; i++)
            {
                #pragma HLS unroll
                for (int j = 0; j < NN; j++)
                {
                    #pragma HLS unroll
                    C[i][j] = 0;
                }
            }

            int ars = (A_R - offset_A_R >= NN) ? NN : A_R - offset_A_R;
            int bcs = (B_C - offset_B_C >= NN) ? NN : B_C - offset_B_C;
            for(int offset_A_C = 0; offset_A_C < A_C; offset_A_C += NN)
            {
                int offset_A = offset_A_R * A_C + offset_A_C;
                int offset_B = offset_A_C * B_C + offset_B_C;
                int acs = (A_C - offset_A_C >= NN) ? NN : A_C - offset_A_C;
                load_data(A0, B0, input_A, input_B, offset_A, offset_B, A_R, A_C, B_C, ars, acs, bcs);
                // PE_array(A0, B0);
                for (int ar = 0; ar < NN / N; ar++)
                {
                    for (int bc=0; bc < NN / N; bc++)
                    {
                        PE_array(A0, B0, ar, bc);
                    }
                }
            }
            
            int offset_C = offset_A_R * B_C + offset_B_C;
            for (int i = 0; i < NN; i++)
            {
                for (int j = 0; j<NN; j++)
                {
                    #pragma HLS pipeline
                    if (i < ars && j < bcs)
                        output_C[offset_C + i * B_C + j] = C[i][j];
                }
            }
        }
    }
}
