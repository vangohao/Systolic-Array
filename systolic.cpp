#include "systolic.h"
#include "hls_stream.h"
#define stmtype hls::stream<dtype>
const int N = 8;
dtype A[N][N];
dtype B[N][N];
dtype C[N][N];

void provide_A(dtype A[N][N], stmtype hpipe[N + 1][N + 1], int i)
{
    for (int k = 0; k<N; k++)
    {
        hpipe[i][0].write(A[i - 1][k]);
    }
}

void provide_B(dtype B[N][N], stmtype vpipe[N + 1][N + 1], int j)
{
    for (int k = 0; k<N; k++)
    {
        vpipe[0][j].write(B[k][j - 1]);
    }
}

void eat_h(stmtype hpipe[N + 1][N + 1], int i)
{
    dtype eat;
    for (int k = 0; k<N; k++)
    {
        eat = hpipe[i][N].read();
    }
}

void eat_v(stmtype vpipe[N + 1][N + 1], int j)
{
    dtype eat;
    for (int k = 0; k<N; k++)
    {
        eat = vpipe[N][j].read();
    }
}

void PE(stmtype hpipe[N + 1][N + 1], stmtype vpipe[N + 1][N + 1], int i, int j)
{
    #pragma HLS stream variable=hpipe dim=1 depth=3
    #pragma HLS stream variable=hpipe dim=2 depth=3
    #pragma HLS stream variable=vpipe dim=1 depth=3
    #pragma HLS stream variable=vpipe dim=2 depth=3

    dtype a;
    dtype b;
    for(int k = 0; k < N; k++)
    {
    // #pragma HLS pipeline ii=1
        // Shift_in(a, b, A, B, hpipe, vpipe, i, j, k);
        a = hpipe[i][j - 1].read();
        b = vpipe[i - 1][j].read();
        C[i - 1][j - 1] += a * b;
        hpipe[i][j].write(a);
        vpipe[i][j].write(b);
        // Shift_out(a, b, hpipe, vpipe, i, j);
    }
}

void PE_array()
{
    #pragma HLS array_partition variable=A complete dim=1
    #pragma HLS array_partition variable=B complete dim=2
    #pragma HLS array_partition variable=C complete dim=1
    #pragma HLS array_partition variable=C complete dim=2

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
        provide_A(A, hpipe, i);
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        provide_B(B, vpipe, i);
    }
    for (int i = 1; i<=N; i++)
    {
        #pragma HLS UNROLL
        for (int j = 1; j<=N; j++)
        {
            #pragma HLS UNROLL
            PE(hpipe, vpipe, i, j);
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
void systolic(dtype *input_A, dtype *input_B, dtype *output_C)
{
    #pragma HLS INTERFACE s_axilite port = return
    #pragma HLS INTERFACE m_axi depth = 256 port = input_A offset = slave
    #pragma HLS INTERFACE m_axi depth = 256 port = input_B offset = slave
    #pragma HLS INTERFACE m_axi depth = 256 port = output_C offset = slave

    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            #pragma HLS pipeline
            A[i][j] = input_A[i * N + j];
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            #pragma HLS pipeline
            B[i][j] = input_B[i * N + j];
        }
    }

    for (int i = 0; i < N; i++)
    {
        #pragma HLS unroll
        for (int j = 0; j<N; j++)
        {
            #pragma HLS unroll
            C[i][j] = 0;
        }
    }

    PE_array();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            #pragma HLS pipeline
            output_C[i * N + j] = C[i][j];
        }
    }
}
