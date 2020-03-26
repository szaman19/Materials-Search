#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PI 3.141592653589793



//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double frac2car_a[], double frac2car_b[], double frac2car_c[],
                double cart_x[], double cart_y[], double cart_z[]);



//expand the lattice to a larger size
void pbc_expand(int *N_atom, int *times_x, int *times_y, int *times_z, double frac_a_frame[], double frac_b_frame[], double frac_c_frame[],
                double epsilon_frame[], double sigma_frame[], double mass_frame[]);


__global__
void cal_Vext(double *V_ext_device, int *a_N_device, int *b_N_device, int *c_N_device, double *cart_x_extended_device, double *cart_y_extended_device, 
        double *cart_z_extended_device, double *cutoff_device, double *epsilon_device, double *sigma_device, int *N_atom_device, int *times_device, 
        int *times_x_device, int *times_y_device, int *times_z_device, double *epsilon_frame_device, double *sigma_frame_device, 
        double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device, double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device);



__global__
void cal_Vext_FH(double *V_ext_device, int *a_N_device, int *b_N_device, int *c_N_device, double *cart_x_extended_device, double *cart_y_extended_device, double *temperature_device,
        double *cart_z_extended_device, double *cutoff_device, double *epsilon_device, double *sigma_device, double *mass_device, int *N_atom_device, int *times_device, 
        int *times_x_device, int *times_y_device, int *times_z_device, double *epsilon_frame_device, double *sigma_frame_device, double *mass_frame_device,
        double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device, double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device);




int main(int argc, char *argv[])
{
    //To calculate the external potential field, two file strings are needed: input filename and output filename
	//define file varaiable
	FILE *fp1;
	int buffersize = 128;
	char str[buffersize];
	//define read-in parameters
	int Nmaxa, Nmaxb, Nmaxc;
	double La, Lb, Lc, dL;
	double alpha, beta, gamma;
    double alpha_rad, beta_rad, gamma_rad;
	double epsilon[1], sigma[1];
    int FH_signal;
    double mass[1], temperature[1];
    int running_block_size;
	double cutoff[1];
	int N_atom[1];
    //define ancillary parameters
    double temp_x[1], temp_y[1], temp_z[1];
    double cart_x, cart_y, cart_z;
    double cart_x_extended[1], cart_y_extended[1], cart_z_extended[1];
    int times_x[1], times_y[1], times_z[1], times[1];
    double a;
    int a_N[1], b_N[1], c_N[1];
    double loc_a, loc_b, loc_c, loc_x, loc_y, loc_z;
    int i, ii, iii, iiii;

    //define metadata parameters - Added by Shehtab 
    char inp_file_name[256];
    double version; 
    char date_buffer[100];
    char time_buffer[100];    
    //done!!!!!

    //read input file parameters
	fp1 = fopen(argv[1], "r");
    fgets(str, buffersize, fp1); // File Name
    fscanf(fp1,"%s\n", inp_file_name);
    // printf("%s\n", inp_file_name); DEBUG

    fgets(str, buffersize, fp1); // Created at
    fscanf(fp1,"%s %s\n", date_buffer,  time_buffer);
    // printf("%s %s\n", date_buffer, time_buffer); DEBUG

    fgets(str, buffersize, fp1); // Version:
    fscanf(fp1,"%lf\n", &version);
    // printf("%lf\n", version); DEBUG

	fgets(str, buffersize, fp1);
	fscanf(fp1,"%d %d %d\n", &Nmaxa, &Nmaxb, &Nmaxc);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf %lf\n", &La, &Lb, &Lc, &dL);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf\n", &alpha, &beta, &gamma);
    alpha_rad = alpha*PI/180;
    beta_rad = beta*PI/180;
    gamma_rad = gamma*PI/180;
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%lf %lf %lf %d %lf %lf %d\n", &epsilon[0], &sigma[0], &cutoff[0], &FH_signal, &mass[0], &temperature[0], &running_block_size);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%d\n", &N_atom[0]);
	fgets(str, buffersize, fp1);
	//done!!!!

    //frac2car parameter calculation
	double frac2car_a[3];
	double frac2car_b[3];
	double frac2car_c[3];
    frac2car_a[0] = La;
    frac2car_a[1] = Lb*cos(gamma_rad);
    frac2car_a[2] = Lc*cos(beta_rad);
    frac2car_b[0] = 0;
    frac2car_b[1] = Lb*sin(gamma_rad);
    frac2car_b[2] = Lc*( (cos(alpha_rad)-cos(beta_rad)*cos(gamma_rad)) / sin(gamma_rad) );
    frac2car_c[2] = La*Lb*Lc*sqrt( 1 - pow(cos(alpha_rad),2) - pow(cos(beta_rad),2) - pow(cos(gamma_rad),2) + 2*cos(alpha_rad)*cos(beta_rad)*cos(gamma_rad) );
	frac2car_c[2] = frac2car_c[2]/(La*Lb*sin(gamma_rad));
	//done!!!!!

    //expand the cell to the size satisfied cutoff condition
    //convert the fractional cell length to cartesian value;
    frac2car(1, 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x = temp_x[0];
    frac2car(0, 1, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y = temp_y[0];
    frac2car(0, 0, 1, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z = temp_z[0];
    times_x[0] = (int) 2*cutoff[0]/cart_x + 1;
    times_y[0] = (int) 2*cutoff[0]/cart_y + 1;
    times_z[0] = (int) 2*cutoff[0]/cart_z + 1;
    times[0] = times_x[0]*times_y[0]*times_z[0];
	double epsilon_frame[N_atom[0]*times[0]], sigma_frame[N_atom[0]*times[0]], mass_frame[N_atom[0]*times[0]];
	double frac_a_frame[N_atom[0]*times[0]], frac_b_frame[N_atom[0]*times[0]], frac_c_frame[N_atom[0]*times[0]];
    for (i=0; i<N_atom[0]; i++)
	{
		fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf\n", &a, &sigma_frame[i], &epsilon_frame[i], &mass_frame[i], &frac_a_frame[i], &frac_b_frame[i], &frac_c_frame[i]);
    	fgets(str, buffersize, fp1);
	}
    fclose(fp1);
    pbc_expand(N_atom, times_x, times_y, times_z, frac_a_frame, frac_b_frame, frac_c_frame, epsilon_frame, sigma_frame, mass_frame);
    frac2car(times_x[0], 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x_extended[0] = temp_x[0];
    frac2car(0, times_y[0], 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y_extended[0] = temp_y[0];
    frac2car(0, 0, times_z[0], frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z_extended[0] = temp_z[0];
    //done!!!!

    

    //calculate the number of the grid
    a_N[0] = La/dL + 1;
	if (a_N[0]>Nmaxa)
	{
		a_N[0] = Nmaxa;
	}
	b_N[0] = Lb/dL + 1;
	if (b_N[0]>Nmaxb)
	{
		b_N[0] = Nmaxb;
	}
	c_N[0] = Lc/dL + 1;
	if (c_N[0]>Nmaxc)
	{
		c_N[0] = Nmaxc;
	}
    //done!!!!!


    

    //define variables on device
    int *a_N_device, *b_N_device, *c_N_device;
    double *cart_x_extended_device, *cart_y_extended_device, *cart_z_extended_device;
    double *cutoff_device;
    double *epsilon_device, *sigma_device, *mass_device, *temperature_device;
    int *N_atom_device;
    int *times_device, *times_x_device, *times_y_device, *times_z_device;
    double *epsilon_frame_device, *sigma_frame_device, *mass_frame_device;
    double *frac_a_frame_device, *frac_b_frame_device, *frac_c_frame_device;
    double *frac2car_a_device, *frac2car_b_device, *frac2car_c_device;
    //allocate memory on device
    cudaMalloc((void **)&a_N_device, sizeof(int));
    cudaMalloc((void **)&b_N_device, sizeof(int));
    cudaMalloc((void **)&c_N_device, sizeof(int));
    cudaMalloc((void **)&cart_x_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_y_extended_device, sizeof(double));
    cudaMalloc((void **)&cart_z_extended_device, sizeof(double));
    cudaMalloc((void **)&cutoff_device, sizeof(double));
    cudaMalloc((void **)&epsilon_device, sizeof(double));
    cudaMalloc((void **)&sigma_device, sizeof(double));
    cudaMalloc((void **)&mass_device, sizeof(double));
    cudaMalloc((void **)&temperature_device, sizeof(double));
    cudaMalloc((void **)&N_atom_device, sizeof(int));
    cudaMalloc((void **)&times_device, sizeof(int));
    cudaMalloc((void **)&times_x_device, sizeof(int));
    cudaMalloc((void **)&times_y_device, sizeof(int));
    cudaMalloc((void **)&times_z_device, sizeof(int));
    cudaMalloc((void **)&epsilon_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&sigma_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&mass_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&frac_a_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&frac_b_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&frac_c_frame_device, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0]);
    cudaMalloc((void **)&frac2car_a_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_b_device, sizeof(double)*3);
    cudaMalloc((void **)&frac2car_c_device, sizeof(double)*3);
    //copy and transfer array
    cudaMemcpy(a_N_device, a_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_N_device, b_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c_N_device, c_N, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cart_x_extended_device, cart_x_extended, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cart_y_extended_device, cart_y_extended, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cart_z_extended_device, cart_z_extended, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cutoff_device, cutoff, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon_device, epsilon, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_device, sigma, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mass_device, mass, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(temperature_device, temperature, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(N_atom_device, N_atom, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(times_device, times, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(times_x_device, times_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(times_y_device, times_y, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(times_z_device, times_z, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(epsilon_frame_device, epsilon_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(sigma_frame_device, sigma_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(mass_frame_device, mass_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(frac_a_frame_device, frac_a_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(frac_b_frame_device, frac_b_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(frac_c_frame_device, frac_c_frame, sizeof(double)*N_atom[0]*times_x[0]*times_y[0]*times_z[0], cudaMemcpyHostToDevice);
    cudaMemcpy(frac2car_a_device, frac2car_a, sizeof(double)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(frac2car_b_device, frac2car_b, sizeof(double)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(frac2car_c_device, frac2car_c, sizeof(double)*3, cudaMemcpyHostToDevice);






    



    double *V_ext, *V_ext_device;
    cudaMallocHost(&V_ext, sizeof(double)*(a_N[0]+1)*(b_N[0]+1)*(c_N[0]+1));
    cudaMalloc((void **)&V_ext_device, sizeof(double)*(a_N[0]+1)*(b_N[0]+1)*(c_N[0]+1));


    


    if (FH_signal==0)
    {
        cal_Vext<<<(int)(((a_N[0]+1)*(b_N[0]+1)*(c_N[0]+1)-1)/running_block_size+1),running_block_size>>>
        (V_ext_device, a_N_device, b_N_device, c_N_device, cart_x_extended_device, cart_y_extended_device, 
            cart_z_extended_device, cutoff_device, epsilon_device, sigma_device, N_atom_device, times_device, 
            times_x_device, times_y_device, times_z_device, epsilon_frame_device, sigma_frame_device, 
            frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, frac2car_a_device, frac2car_b_device, frac2car_c_device);
    }
    else if (FH_signal==1)
    {
        cal_Vext_FH<<<(int)(((a_N[0]+1)*(b_N[0]+1)*(c_N[0]+1)-1)/running_block_size+1),running_block_size>>>
        (V_ext_device, a_N_device, b_N_device, c_N_device, cart_x_extended_device, cart_y_extended_device, temperature_device,
            cart_z_extended_device, cutoff_device, epsilon_device, sigma_device, mass_device, N_atom_device, times_device, 
            times_x_device, times_y_device, times_z_device, epsilon_frame_device, sigma_frame_device, mass_frame_device,
            frac_a_frame_device, frac_b_frame_device, frac_c_frame_device, frac2car_a_device, frac2car_b_device, frac2car_c_device);
    }
    
    


    //output the energy profiles
    fp1 = fopen(argv[2],"w+");
    cudaMemcpy(V_ext, V_ext_device, sizeof(double)*(a_N[0]+1)*(b_N[0]+1)*(c_N[0]+1), cudaMemcpyDeviceToHost);
    fprintf(fp1, "File Name: %s\n", argv[2]);
    time_t timer;
    char buffer[26];
    struct tm* tm_info;
    timer = time(NULL);
    tm_info = localtime(&timer);
    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    fprintf(fp1, "Created at %s :\n", buffer);
    fprintf(fp1, "Version: %lf\n", 1.1);


    fprintf(fp1, "Input File Name: %s\n", inp_file_name);
    fprintf(fp1, "Input File Created: %s %s\n",date_buffer, time_buffer);
    fprintf(fp1, "Input File Generator Version: %lf\n", version);

    fprintf(fp1,"%d\t%d\t%d\n", Nmaxa, Nmaxb, Nmaxc);
    fprintf(fp1,"La Lb Lc dL\n");
    fprintf(fp1,"%lf\t%lf\t%lf\t%lf\n", La, Lb, Lc, dL);
    fprintf(fp1,"Alpha Beta Gamma\n");
    fprintf(fp1,"%lf\t%lf\t%lf\n", alpha, beta, gamma);
    fprintf(fp1,"Epsilon(K) Sigma(A) cutoff(A) FH_signal mass(g/mol) Tempearture(K)\n");
    fprintf(fp1,"%lf\t%lf\t%lf\t%d\t%lf\t%lf\n", epsilon, sigma, cutoff, FH_signal, mass[0], temperature[0]);
    fprintf(fp1,"Number of atoms\n");
    fprintf(fp1,"%d\n", N_atom[0]); 


    iiii = 0;
    for (i=0; i<=a_N[0]; i++)
    {
        loc_a = i*(1.0/a_N[0]);
        for (ii=0; ii<=b_N[0]; ii++)
        {
            loc_b = ii*(1.0/b_N[0]);
            for (iii=0; iii<=c_N[0]; iii++)
            {
                loc_c = iii*(1.0/c_N[0]);
                frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
                loc_x = temp_x[0];
                loc_y = temp_y[0];
                loc_z = temp_z[0];
                fprintf(fp1, "%lf\t%lf\t%lf\t%lf\n", loc_x, loc_y, loc_z, V_ext[iiii]);
                iiii++;
            }
        }
        
    }
    fclose(fp1);
}





















//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double frac2car_a[], double frac2car_b[], double frac2car_c[],
                double cart_x[], double cart_y[], double cart_z[])
{
    cart_x[0] = frac_a*frac2car_a[0] + frac_b*frac2car_a[1] + frac_c*frac2car_a[2];
    cart_y[0] = frac_a*frac2car_b[0] + frac_b*frac2car_b[1] + frac_c*frac2car_b[2];
    cart_z[0] = frac_a*frac2car_c[0] + frac_b*frac2car_c[1] + frac_c*frac2car_c[2];
}



//expand the lattice to a larger size
void pbc_expand(int *N_atom, int *times_x, int *times_y, int *times_z, double frac_a_frame[], double frac_b_frame[], double frac_c_frame[],
                double epsilon_frame[], double sigma_frame[], double mass_frame[])
{
    int i, ii, iii, iiii;
    int j;
    iiii = 0;
    for (j=0; j<N_atom[0]; j++)
    {
        for (i=0; i<times_x[0]; i++)
        {
            for (ii=0; ii<times_y[0]; ii++)
            {
                for (iii=0; iii<times_z[0]; iii++)
                {
                    if ((i!=0)||(ii!=0)||(iii!=0))
                    {
                        frac_a_frame[N_atom[0]+iiii] = frac_a_frame[j] + i;
                        frac_b_frame[N_atom[0]+iiii] = frac_b_frame[j] + ii;
                        frac_c_frame[N_atom[0]+iiii] = frac_c_frame[j] + iii;
                        epsilon_frame[N_atom[0]+iiii] = epsilon_frame[j];
                        sigma_frame[N_atom[0]+iiii] = sigma_frame[j];
                        mass_frame[N_atom[0]+iiii] = mass_frame[j];
                        iiii++;
                    }
                }
            }
        }
    }
}



//calcualte external potenial
__global__
void cal_Vext(double *V_ext_device, int *a_N_device, int *b_N_device, int *c_N_device, double *cart_x_extended_device, double *cart_y_extended_device, 
        double *cart_z_extended_device, double *cutoff_device, double *epsilon_device, double *sigma_device, int *N_atom_device, int *times_device, 
        int *times_x_device, int *times_y_device, int *times_z_device, double *epsilon_frame_device, double *sigma_frame_device, 
        double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device, double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    int index_a, index_b, index_c;
    double pos_a, pos_b, pos_c;
    double delta_a, delta_b, delta_c;
    double delta_x, delta_y, delta_z;
    double dis;
    double epsilon_cal, sigma_cal;
    double shift;

    for (i=index; i<(a_N_device[0]*b_N_device[0]*c_N_device[0]); i+=stride)
    {
        index_a = (int) ( i/(b_N_device[0]*c_N_device[0]) );
        index_b = (int) ( (i-index_a*b_N_device[0]*c_N_device[0])/c_N_device[0] );
        index_c = (int) ( i-index_a*b_N_device[0]*c_N_device[0]-index_b*c_N_device[0] );

        V_ext_device[i] = 0;

        pos_a = index_a * 1.0 / a_N_device[0];
        pos_b = index_b * 1.0 / b_N_device[0];
        pos_c = index_c * 1.0 / c_N_device[0];

        for (ii=0; ii<N_atom_device[0]*times_device[0]; ii++)
        {
            delta_a = pos_a - frac_a_frame_device[ii];
            delta_b = pos_b - frac_b_frame_device[ii];
            delta_c = pos_c - frac_c_frame_device[ii];

            delta_x = delta_a*frac2car_a_device[0] + delta_b*frac2car_a_device[1] + delta_c*frac2car_a_device[2];
            delta_y = delta_a*frac2car_b_device[0] + delta_b*frac2car_b_device[1] + delta_c*frac2car_b_device[2];
            delta_z = delta_a*frac2car_c_device[0] + delta_b*frac2car_c_device[1] + delta_c*frac2car_c_device[2];

            if (delta_x > 1.0*cart_x_extended_device[0]/2)
            {
                delta_a = delta_a - times_x_device[0];
            }
            else if (delta_x < -1.0*cart_x_extended_device[0]/2)
            {
                delta_a = delta_a + times_x_device[0];
            }

            if (delta_y > 1.0*cart_y_extended_device[0]/2)
            {
                delta_b = delta_b - times_y_device[0];
            }
            else if (delta_y < -1.0*cart_y_extended_device[0]/2)
            {
                delta_b = delta_b + times_y_device[0];
            }

            if (delta_z > 1.0*cart_z_extended_device[0]/2)
            {
                delta_c = delta_c - times_z_device[0];
            }
            else if (delta_z < -1.0*cart_z_extended_device[0]/2)
            {
                delta_c = delta_c + times_z_device[0];
            }

            delta_x = delta_a*frac2car_a_device[0] + delta_b*frac2car_a_device[1] + delta_c*frac2car_a_device[2];
            delta_y = delta_a*frac2car_b_device[0] + delta_b*frac2car_b_device[1] + delta_c*frac2car_b_device[2];
            delta_z = delta_a*frac2car_c_device[0] + delta_b*frac2car_c_device[1] + delta_c*frac2car_c_device[2];

            dis = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

            if (dis<cutoff_device[0])
            {
                epsilon_cal = sqrt(epsilon_device[0]*epsilon_frame_device[ii]);
                sigma_cal = 0.5*(sigma_device[0]+sigma_frame_device[ii]);

                shift = 4*epsilon_cal*( pow((sigma_cal/cutoff_device[0]),12)-pow((sigma_cal/cutoff_device[0]),6) );

                if (dis<0.1*sigma_cal)
                {
                    dis = 0.1*sigma_cal;
                }

                V_ext_device[i] = V_ext_device[i] + 4*epsilon_cal*( pow((sigma_cal/dis),12)-pow((sigma_cal/dis),6) ) 
                - shift;
            }
        }
    }
}



//calculate external potenial based on FH approximation
__global__
void cal_Vext_FH(double *V_ext_device, int *a_N_device, int *b_N_device, int *c_N_device, double *cart_x_extended_device, double *cart_y_extended_device, double *temperature_device,
        double *cart_z_extended_device, double *cutoff_device, double *epsilon_device, double *sigma_device, double *mass_device, int *N_atom_device, int *times_device, 
        int *times_x_device, int *times_y_device, int *times_z_device, double *epsilon_frame_device, double *sigma_frame_device, double *mass_frame_device,
        double *frac_a_frame_device, double *frac_b_frame_device, double *frac_c_frame_device, double *frac2car_a_device, double *frac2car_b_device, double *frac2car_c_device)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int i, ii;
    int index_a, index_b, index_c;
    double pos_a, pos_b, pos_c;
    double delta_a, delta_b, delta_c;
    double delta_x, delta_y, delta_z;
    double dis;
    double epsilon_cal, sigma_cal;
    double shift;

    //variables used for FH aprroximations
    double plank_constant = 6.626070040e-34;                                // J*s
    double h_bar = plank_constant/(2*PI);                                   // J*s
    double k = 1.38064852e-23;                                              // Boltzmann constant: J/K
    double mass1, mass2;                                                    //
    double beta = 1/(k*temperature_device[0]);                              // 1/J
    double NA = 6.023e23;                                                   // Avogadro constant
    double mu;
    double C1, C2;
    double lj_potential;
    mass1 = mass_device[0]/1e3/NA;

    //calculate the external potential
    for (i=index; i<((a_N_device[0]+1)*(a_N_device[0]+1)*(a_N_device[0]+1)); i+=stride)
    {
        index_a = (int) ( i/((b_N_device[0]+1)*(c_N_device[0]+1)) );
        index_b = (int) ( (i-index_a*(b_N_device[0]+1)*(c_N_device[0]+1))/(c_N_device[0]+1) );
        index_c = (int) ( i-index_a*(b_N_device[0]+1)*(c_N_device[0]+1)-index_b*(c_N_device[0]+1));

        V_ext_device[i] = 0;

        pos_a = index_a * 1.0 / a_N_device[0];
        pos_b = index_b * 1.0 / b_N_device[0];
        pos_c = index_c * 1.0 / c_N_device[0];

        for (ii=0; ii<N_atom_device[0]*times_device[0]; ii++)
        {
            mass2 = mass_frame_device[ii]/1e3/NA;

            delta_a = pos_a - frac_a_frame_device[ii];
            delta_b = pos_b - frac_b_frame_device[ii];
            delta_c = pos_c - frac_c_frame_device[ii];

            delta_x = delta_a*frac2car_a_device[0] + delta_b*frac2car_a_device[1] + delta_c*frac2car_a_device[2];
            delta_y = delta_a*frac2car_b_device[0] + delta_b*frac2car_b_device[1] + delta_c*frac2car_b_device[2];
            delta_z = delta_a*frac2car_c_device[0] + delta_b*frac2car_c_device[1] + delta_c*frac2car_c_device[2];

            if (delta_x > 1.0*cart_x_extended_device[0]/2)
            {
                delta_a = delta_a - times_x_device[0];
            }
            else if (delta_x < -1.0*cart_x_extended_device[0]/2)
            {
                delta_a = delta_a + times_x_device[0];
            }

            if (delta_y > 1.0*cart_y_extended_device[0]/2)
            {
                delta_b = delta_b - times_y_device[0];
            }
            else if (delta_y < -1.0*cart_y_extended_device[0]/2)
            {
                delta_b = delta_b + times_y_device[0];
            }

            if (delta_z > 1.0*cart_z_extended_device[0]/2)
            {
                delta_c = delta_c - times_z_device[0];
            }
            else if (delta_z < -1.0*cart_z_extended_device[0]/2)
            {
                delta_c = delta_c + times_z_device[0];
            }

            delta_x = delta_a*frac2car_a_device[0] + delta_b*frac2car_a_device[1] + delta_c*frac2car_a_device[2];
            delta_y = delta_a*frac2car_b_device[0] + delta_b*frac2car_b_device[1] + delta_c*frac2car_b_device[2];
            delta_z = delta_a*frac2car_c_device[0] + delta_b*frac2car_c_device[1] + delta_c*frac2car_c_device[2];

            dis = sqrt(pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2));

            if (dis<cutoff_device[0])
            {
                epsilon_cal = sqrt(epsilon_device[0]*epsilon_frame_device[ii]);
                sigma_cal = 0.5*(sigma_device[0]+sigma_frame_device[ii]);

                mu = (mass1*mass2)/(mass1+mass2);

                //If h_bar is in the unit of J*s, beta is in the unit of 1/J, mass/mu is in the unit of kg,
                //Then the unit of the constant C1 is m2
                C1 = beta*pow(h_bar,2)/(24*mu);
                //However, since the length unit used in potential calcualtion is angstrom, the unit of constant needs to be done
                C1 = C1*1e20;
                //If h_bar is in the unit of J*s, beta is in the unit of 1/J, mass/mu is in the unit of kg,
                //Then the unit of the constant C1 is m4
                C2 = pow(beta,2)*pow(h_bar,4)/(1152*pow(mu,2));
                //However, since the length unit used in potential calcualtion is angstrom, the unit of constant needs to be done
                C2 = C2*1e40;

                shift = 4*epsilon_cal* ( pow(sigma_cal,12)/pow(cutoff_device[0],12) - pow(sigma_cal,6)/pow(cutoff_device[0],6) ) +
                        C1*4*epsilon_cal* ( 132*pow(sigma_cal,12)/pow(cutoff_device[0],14) - 30*pow(sigma_cal,6)/pow(cutoff_device[0],8) ) +
                        C2*4*epsilon_cal* ( 23844*pow(sigma_cal,12)/pow(cutoff_device[0],16) - 1590*pow(sigma_cal,6)/pow(cutoff_device[0],10) );

                if (dis<0.1*sigma_cal)
                {
                    dis = 0.1*sigma_cal;
                }

                lj_potential = 4*epsilon_cal* ( pow(sigma_cal,12)/pow(dis,12) - pow(sigma_cal,6)/pow(dis,6) ) +
                        C1*4*epsilon_cal* ( 132*pow(sigma_cal,12)/pow(dis,14) - 30*pow(sigma_cal,6)/pow(dis,8) ) +
                        C2*4*epsilon_cal* ( 23844*pow(sigma_cal,12)/pow(dis,16) - 1590*pow(sigma_cal,6)/pow(dis,10) );

                V_ext_device[i] = V_ext_device[i] + lj_potential - shift;
            }
        }
    }
}
