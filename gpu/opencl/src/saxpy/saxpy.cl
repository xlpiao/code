__kernel void saxpy(float alpha,
                    __global float *x,
                    __global float *y,
                    __global float *z,
                    int n)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	if ((i < n) && (j < n))
	{
    z[i * n + j] = alpha * x[i * n + j] + y[i * n + j];
	}
}
