#include "systolic.h"
#include <iostream>
#include "hls_stream.h"
#define stmtype hls::stream<dtype>
const int b_AR = 260;
const int b_AC = 117;
const int b_BC = 65;
const int N = 13;
dtype A0[b_AR][b_AC];
dtype B0[b_AC][b_BC];
dtype A1[b_AR][b_AC];
dtype B1[b_AC][b_BC];
dtype C[b_AR][b_BC];
dtype C0[N][N];

void provide_A(dtype A[b_AR][b_AC], stmtype hpipe[N + 1][N + 1], int i, int ar)
{
    for (int k = 0; k<b_AC; k++)
    {
        #pragma HLS pipeline
        hpipe[i][0].write(A[ar * N + i - 1][k]);
    }
}

void provide_B(dtype B[b_AC][b_BC], stmtype vpipe[N + 1][N + 1], int j, int bc)
{
    for (int k = 0; k<b_AC; k++)
    {
        #pragma HLS pipeline
        vpipe[0][j].write(B[k][bc * N + j - 1]);
    }
}

void eat_h(stmtype hpipe[N + 1][N + 1], int i)
{
    dtype eat;
    for (int k = 0; k<b_AC; k++)
    {
        #pragma HLS pipeline
        eat = hpipe[i][N].read();
    }
}

void eat_v(stmtype vpipe[N + 1][N + 1], int j)
{
    dtype eat;
    for (int k = 0; k<b_AC; k++)
    {
        #pragma HLS pipeline
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
    int x = ar * N + i - 1;
    int y = bc * N + j - 1;
    for(int k = 0; k < b_AC; k++)
    {
    // #pragma HLS latency min = 4 max = 4
    #pragma HLS pipeline
        // Shift_in(a, b, A, B, hpipe, vpipe, i, j, k);
        a = hpipe[i][j - 1].read();
        b = vpipe[i - 1][j].read();
        // C[x][y] += a * b;
        C0[i - 1][j - 1] += a * b;
        hpipe[i][j].write(a);
        vpipe[i][j].write(b);
        // Shift_out(a, b, hpipe, vpipe, i, j);
    }
}

void PE_array(dtype A[b_AR][b_AC], dtype B[b_AC][b_BC], int ar, int bc)
{
    #pragma HLS array_partition variable=A cyclic factor=N dim=1
    #pragma HLS array_partition variable=B cyclic factor=N dim=2
    // #pragma HLS array_partition variable=C cyclic factor=N dim=1
    // #pragma HLS array_partition variable=C cyclic factor=N dim=2
    #pragma HLS array_partition variable=C0 complete dim=1
    #pragma HLS array_partition variable=C0 complete dim=2

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

void load_data(dtype A[b_AR][b_AC], dtype B[b_AC][b_BC], dtype *input_A, dtype *input_B, int offsetA, int offsetB, int A_R, int A_C, int B_C, int ars, int acs, int bcs)
{
    for (int i = 0; i < b_AR; i++)
    {
        for (int j = 0; j < b_AC; j++)
        {
            #pragma HLS pipeline
            if (i < ars && j < acs)
                A[i][j] = input_A[offsetA + i * A_C + j];
            else
                A[i][j] = 0;
        }
    }

    for (int i = 0; i < b_AC; i++)
    {
        for (int j = 0; j<b_BC; j++)
        {
            #pragma HLS pipeline
            if (i < acs && j < bcs)
                B[i][j] = input_B[offsetB + i * B_C + j];
            else
                B[i][j] = 0;
        }
    }
}

void calc(dtype A0[b_AR][b_AC], dtype B0[b_AC][b_BC], int offset)
{
    #pragma HLS array_partition variable=A0 cyclic factor=N dim=1
    #pragma HLS array_partition variable=A1 cyclic factor=N dim=1
    #pragma HLS array_partition variable=B0 cyclic factor=N dim=2
    #pragma HLS array_partition variable=B1 cyclic factor=N dim=2
    #pragma HLS array_partition variable=C cyclic factor=N dim=1
    #pragma HLS array_partition variable=C cyclic factor=N dim=2
    #pragma HLS array_partition variable=C0 complete dim=1
    #pragma HLS array_partition variable=C0 complete dim=2
    if (offset)
    {
        for (int ar = 0; ar < b_AR / N; ar++)
        {
            for (int bc=0; bc < b_BC / N; bc++)
            {
                for (int i = 0; i < N; i++)
                {
                    #pragma HLS unroll
                    for (int j = 0; j < N; j++)
                    {
                        #pragma HLS unroll
                        C0[i][j] = C[ar * N + i][bc * N + j];
                    }
                }
                PE_array(A0, B0, ar, bc);
                for (int i = 0; i < N; i++)
                {
                    #pragma HLS unroll
                    for (int j = 0; j < N; j++)
                    {
                        #pragma HLS unroll
                        C[ar * N + i][bc * N + j] = C0[i][j];
                    }
                }
            }
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
    #pragma HLS array_partition variable=C0 complete dim=1
    #pragma HLS array_partition variable=C0 complete dim=2
    // #pragma HLS resource variable=C0 core=RAM_1P_LUTRAM

    for(int offset_A_R = 0; offset_A_R < A_R; offset_A_R += b_AR)
    {
        for(int offset_B_C = 0; offset_B_C < B_C; offset_B_C += b_BC)
        {
            // std::cerr<<offset_A_R<<","<<offset_B_C<<std::endl;
            for (int i = 0; i < b_AR; i++)
            {
                #pragma HLS unroll
                for (int j = 0; j < b_BC; j++)
                {
                    #pragma HLS unroll
                    C[i][j] = 0;
                }
            }

            int ars = (A_R - offset_A_R >= b_AR) ? b_AR : A_R - offset_A_R;
            int bcs = (B_C - offset_B_C >= b_BC) ? b_BC : B_C - offset_B_C;
            bool ping_pong_flag = 1;
            for(int offset_A_C = 0; offset_A_C < A_C + b_AC; offset_A_C += b_AC)
            {
                int offset_A = offset_A_R * A_C + offset_A_C;
                int offset_B = offset_A_C * B_C + offset_B_C;
                int acs = (A_C - offset_A_C >= b_AC) ? b_AC : A_C - offset_A_C;
                if (ping_pong_flag)
                {
                    load_data(A0, B0, input_A, input_B, offset_A, offset_B, A_R, A_C, B_C, ars, acs, bcs);
                    calc(A1, B1, offset_A_C);
                }
                else
                {
                    load_data(A1, B1, input_A, input_B, offset_A, offset_B, A_R, A_C, B_C, ars, acs, bcs);
                    calc(A0, B0, offset_A_C);
                }
                ping_pong_flag = !ping_pong_flag;
            }
            
            int offset_C = offset_A_R * B_C + offset_B_C;
            for (int i = 0; i < b_AR; i++)
            {
                for (int j = 0; j<b_BC; j++)
                {
                    #pragma HLS pipeline
                    if (i < ars && j < bcs)
                        output_C[offset_C + i * B_C + j] = C[i][j];
                }
            }
        }
    }
}
