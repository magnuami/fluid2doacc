// float4 datatype
struct float4 {
	float x, y, z, w;
};

typedef struct float4 float4;

// struct used for fluid simulation data
struct Configuration {
	int N, size; // grid dimension

	float *velx, *velx0; // velocity in x direction
	float *vely, *vely0; // velocity in y direction
	float *dens, *dens0; // density
	float *pres, *pres0; // pressure
	float *div;			 // divergence
};


