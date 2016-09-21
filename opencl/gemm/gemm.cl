__kernel void gemm(__global float *a, __global float *b, __global float *c, float alpha, float beta, int n)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if ((i < n) && (j < n))
	{	
		c[i * n + j] *= beta;
		for(int k=0; k < n; k++)
		{
			c[i * n + j] += alpha * a[i * n + k] * b[k * n +j];
		}
	}
}
