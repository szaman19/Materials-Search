#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PI 3.141592653589793


//convert fractional coordinate to cartesian coordinate
void frac2car(double frac_a, double frac_b, double frac_c, double frac2car_a[], double frac2car_b[], double frac2car_c[],
                double cart_x[], double cart_y[], double cart_z[]);



//expand the lattice to a larger size
void pbc_expand(int N_atom, int times_x, int times_y, int times_z, double frac_a_frame[], double frac_b_frame[], double frac_c_frame[],
                double epsilon_frame[], double sigma_frame[], double mass_frame[]);

//calculate the distance of two point
double calc_dis(double loc_x, double loc_y, double loc_z, double temp_frame_x, double temp_frame_y, double temp_frame_z);

//calculate possible minimum (cartesian) distance by considering periodic boundary condition
double calc_minimum_dis(double loc_x, double loc_y, double loc_z, double temp_frame_a, double temp_frame_b, double temp_frame_c,
                        double frac2car_a[], double frac2car_b[], double frac2car_c[], double cutoff, int times_x, int times_y, int times_z,
                        double alpha, double beta, double gamma, double La, double Lb, double Lc, double cart_x, double cart_y, double cart_z);

//calculate pure LJ(12-6) potentialat that point
double calc_pure_lj(double dis, double sigma1, double sigma2, double epsilon1, double epsilon2);

//calculate LJ potential with shift at that point
double calc_lj(double dis, double cutoff, double sigma, double sigma_frame_iiii, double epsilon, double epsilon_frame_iiii);

//calculate Feynman-Hibbs 4th order corrected LJ potentialat that point
double calc_pure_FH_lj(double dis, double sigma1, double sigma2, double epsilon1, double epsilon2, double mass1, double mass2,
                        double h_bar, double beta);

//calculate Feynman-Hibbs 4th order corrected LJ potential with shift at that point
double calc_FH_lj(double dis, double cutoff, double sigma, double sigma_frame_iiii, double epsilon, double epsilon_frame_iiii,
                    double mass1, double mass2, double h_bar, double beta);


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
	double epsilon, sigma;
    int FH_signal;
    double mass, temperature;
	double cutoff;
	int N_atom;
    //define ancillary parameters
    double temp_x[1], temp_y[1], temp_z[1];
    double cart_x, cart_y, cart_z;
    double cart_x_extended, cart_y_extended, cart_z_extended;
    int times_x, times_y, times_z, times;
    double a;
    int a_N, b_N, c_N;
    double shift;
    double loc_a, loc_b, loc_c, loc_x, loc_y, loc_z, loc_u;
    double temp_frame_a, temp_frame_b, temp_frame_c;
    double temp_u;
    int i, ii, iii, iiii;
    double dis;
    //done!!!!!

    //read input file parameters
	fp1 = fopen(argv[1], "r");
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
	fscanf(fp1,"%lf %lf %lf %d %lf %lf\n", &epsilon, &sigma, &cutoff, &FH_signal, &mass, &temperature);
	fgets(str, buffersize, fp1);
	fscanf(fp1,"%d\n", &N_atom);
	fgets(str, buffersize, fp1);
	//done!!!!

    //test memory
	double *test_memory;
	int temp_size = Nmaxa*Nmaxb*Nmaxc;
	test_memory = malloc(temp_size*sizeof(double));
	free(test_memory);
	//done!!!!!

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
    times_x = (int) 2*cutoff/cart_x + 1;
    times_y = (int) 2*cutoff/cart_y + 1;
    times_z = (int) 2*cutoff/cart_z + 1;
    times = times_x*times_y*times_z;
	double epsilon_frame[N_atom*times], sigma_frame[N_atom*times], mass_frame[N_atom*times];
	double frac_a_frame[N_atom*times], frac_b_frame[N_atom*times], frac_c_frame[N_atom*times];
    for (i=0; i<N_atom; i++)
	{
		fscanf(fp1,"%lf %lf %lf %lf %lf %lf %lf\n", &a, &sigma_frame[i], &epsilon_frame[i], &mass_frame[i], &frac_a_frame[i], &frac_b_frame[i], &frac_c_frame[i]);
    	fgets(str, buffersize, fp1);
	}
    fclose(fp1);
    pbc_expand(N_atom, times_x, times_y, times_z, frac_a_frame, frac_b_frame, frac_c_frame, epsilon_frame, sigma_frame, mass_frame);
    frac2car(times_x, 0, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_x_extended = temp_x[0];
    frac2car(0, times_y, 0, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_y_extended = temp_y[0];
    frac2car(0, 0, times_z, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    cart_z_extended = temp_z[0];
    //done!!!!

    //calculate the number of the grid
    a_N = La/dL + 1;
	if (a_N>Nmaxa)
	{
		a_N = Nmaxa;
	}
	b_N = Lb/dL + 1;
	if (b_N>Nmaxb)
	{
		b_N = Nmaxb;
	}
	c_N = Lc/dL + 1;
	if (c_N>Nmaxc)
	{
		c_N = Nmaxc;
	}
    //done!!!!!
    //external potential calculation
    fp1 = fopen(argv[2],"w+");

    /*
    Add versioning here 
    */

    if (FH_signal==0)
    {
        //Feynman-Hibbs approximation is turned cutoff
        //Use regular LJ potential
        for (i=0; i<=a_N; i++)
    	{
    		loc_a = i*(1.0/a_N);
    		for (ii=0; ii<=b_N; ii++)
    		{
    			loc_b = ii*(1.0/b_N);
    			for (iii=0; iii<=c_N; iii++)
    			{
                    loc_c = iii*(1.0/c_N);
                    loc_u = 0;
                    for (iiii=0; iiii<N_atom*times; iiii++)
                    {
                        temp_frame_a = frac_a_frame[iiii];
                        temp_frame_b = frac_b_frame[iiii];
                        temp_frame_c = frac_c_frame[iiii];
                        frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
                        loc_x = temp_x[0];
                        loc_y = temp_y[0];
                        loc_z = temp_z[0];
                        dis = calc_minimum_dis(loc_x, loc_y, loc_z, temp_frame_a, temp_frame_b, temp_frame_c, frac2car_a, frac2car_b, frac2car_c, cutoff,
                                                times_x, times_y, times_z, alpha, beta, gamma, La, Lb, Lc, cart_x_extended, cart_y_extended, cart_z_extended);
                        temp_u = calc_lj(dis, cutoff, sigma, sigma_frame[iiii], epsilon, epsilon_frame[iiii]);
                        loc_u = loc_u + temp_u;
                    }
                    frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
                    loc_x = temp_x[0];
                    loc_y = temp_y[0];
                    loc_z = temp_z[0];
                    fprintf(fp1, "%lf\t%lf\t%lf\t%lf\n", loc_x, loc_y, loc_z, loc_u);
    			}
    		}
    	}
    }
    else if (FH_signal==1)
    {
        //Feynman-Hibbs approximation is turned on
        //Use Feynman-Hibbs corrected LJ potential
        //define parameters needed for FH LJ potential;
        double plank_constant = 6.626070040e-34;                                // J*s
        double h_bar = plank_constant/(2*PI);                                   // J*s
        double k = 1.38064852e-23;                                              // Boltzmann constant: J/K
        double mass1, mass2;                                                    // g/mole
        double beta = 1/(k*temperature);                                        // 1/J
        double NA = 6.023e23;                                                   // Avogadro constant
        mass1 = mass/1e3/NA;
        for (i=0; i<=a_N; i++)
    	{
    		loc_a = i*(1.0/a_N);
    		for (ii=0; ii<=b_N; ii++)
    		{
    			loc_b = ii*(1.0/b_N);
    			for (iii=0; iii<=c_N; iii++)
    			{
                    loc_c = iii*(1.0/c_N);
                    loc_u = 0;
                    for (iiii=0; iiii<N_atom*times; iiii++)
                    {
                        mass2 = mass_frame[iiii]/1e3/NA;
                        temp_frame_a = frac_a_frame[iiii];
                        temp_frame_b = frac_b_frame[iiii];
                        temp_frame_c = frac_c_frame[iiii];
                        frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
                        loc_x = temp_x[0];
                        loc_y = temp_y[0];
                        loc_z = temp_z[0];
                        dis = calc_minimum_dis(loc_x, loc_y, loc_z, temp_frame_a, temp_frame_b, temp_frame_c, frac2car_a, frac2car_b, frac2car_c, cutoff,
                                                times_x, times_y, times_z, alpha, beta, gamma, La, Lb, Lc, cart_x_extended, cart_y_extended, cart_z_extended);
                        temp_u = calc_FH_lj(dis, cutoff, sigma, sigma_frame[iiii], epsilon, epsilon_frame[iiii], mass1, mass2, h_bar, beta);
                        loc_u = loc_u + temp_u;
                    }
                    frac2car(loc_a, loc_b, loc_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
                    loc_x = temp_x[0];
                    loc_y = temp_y[0];
                    loc_z = temp_z[0];
                    fprintf(fp1, "%lf\t%lf\t%lf\t%lf\n", loc_x, loc_y, loc_z, loc_u);
    			}
    		}
    	}
    }
    else
    {
        printf("FATAL ERROR!!!!!!!!! Wrong FH_sinal input\n");
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
void pbc_expand(int N_atom, int times_x, int times_y, int times_z, double frac_a_frame[], double frac_b_frame[], double frac_c_frame[],
                double epsilon_frame[], double sigma_frame[], double mass_frame[])
{
	int i, ii, iii, iiii;
	int j;
	iiii = 0;
	for (j=0; j<N_atom; j++)
	{
		for (i=0; i<times_x; i++)
		{
			for (ii=0; ii<times_y; ii++)
			{
				for (iii=0; iii<times_z; iii++)
				{
					if ((i!=0)||(ii!=0)||(iii!=0))
					{
						frac_a_frame[N_atom+iiii] = frac_a_frame[j] + i;
						frac_b_frame[N_atom+iiii] = frac_b_frame[j] + ii;
						frac_c_frame[N_atom+iiii] = frac_c_frame[j] + iii;
						epsilon_frame[N_atom+iiii] = epsilon_frame[j];
						sigma_frame[N_atom+iiii] = sigma_frame[j];
						mass_frame[N_atom+iiii] = mass_frame[j];
						iiii++;
					}
				}
			}
		}
	}
}



//calculate the distance of two point
double calc_dis(double loc_x, double loc_y, double loc_z, double temp_frame_x, double temp_frame_y, double temp_frame_z)
{
    double distance;
    distance = sqrt(pow((loc_x-temp_frame_x),2)+pow((loc_y-temp_frame_y),2)+pow((loc_z-temp_frame_z),2));
    return distance;
}



//calculate possible minimum (cartesian) distance by considering periodic boundary condition
double calc_minimum_dis(double loc_x, double loc_y, double loc_z, double temp_frame_a, double temp_frame_b, double temp_frame_c,
                        double frac2car_a[], double frac2car_b[], double frac2car_c[], double cutoff, int times_x, int times_y, int times_z,
                        double alpha, double beta, double gamma, double La, double Lb, double Lc, double cart_x, double cart_y, double cart_z)
{
    double temp_x[1], temp_y[1], temp_z[1];
    double temp_frame_x, temp_frame_y, temp_frame_z;
    double modify_frame_a=0, modify_frame_b=0, modify_frame_c=0;
    double delta_x, delta_y, delta_z;
    int i, ii, iii;
    double distance, temp_distance;

    frac2car(temp_frame_a, temp_frame_b, temp_frame_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    temp_frame_x = temp_x[0];
    temp_frame_y = temp_y[0];
    temp_frame_z = temp_z[0];

    //modifty the fractional coordinate based on periodic boundary condition
    //z-direction
    delta_z = loc_z - temp_frame_z;
    if (delta_z > cart_z/2)
    {
        //local point is higher than the atom in z direction over half lattice length
        //lift the atom by one periodic box size in z direction
        modify_frame_c = temp_frame_c+times_z;
    }
    else if (delta_z < -cart_z/2)
    {
        //local point is lower than the atom in z direction over half lattice length
        //lower the atom by one periodic box size in z direction
        modify_frame_c = temp_frame_c-times_z;
    }
    else
    {
        //local point is similar height as the atom in z direction
        //nothing needs to be changed in z direction
        modify_frame_c = temp_frame_c;
    }
    //y-direction
    frac2car(temp_frame_a, temp_frame_b, modify_frame_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    temp_frame_y = temp_y[0];
    delta_y = loc_y - temp_frame_y;
    if (delta_y > cart_y/2)
    {
        //local point is higher than atom in y direction over half lattice length
        //lift the atom by one periodic box size in y direction
        modify_frame_b = temp_frame_b+times_y;
    }
    else if (delta_y < -cart_y/2)
    {
        //local point is lower than the atom in y direction over half lattice length
        //lower the atom by one periodic box size in y direction
        modify_frame_b = temp_frame_b-times_y;
    }
    else
    {
        //local point is similar height as the atom in y direction
        //nothing needs to be changed in y direction
        modify_frame_b = temp_frame_b;
    }
    //x-direction
    frac2car(temp_frame_a, modify_frame_b, modify_frame_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    temp_frame_x = temp_x[0];
    delta_x = loc_x - temp_frame_x;
    if (delta_x > cart_x/2)
    {
        //local point is higher than atom in x direction over half lattice length
        //lift the atom by one periodic box size in x direction
        modify_frame_a = temp_frame_a+times_x;
    }
    else if (delta_x < -cart_x/2)
    {
        //local point is lower than the atom in x direction over half lattice length
        //lower the atom by one periodic box size in x direction
        modify_frame_a = temp_frame_a-times_x;
    }
    else
    {
        //local point is similar height as the atom in x direction
        //nothing needs to be changed in x direction
        modify_frame_a = temp_frame_a;
    }
    //done!!!!!

    frac2car(modify_frame_a, modify_frame_b, modify_frame_c, frac2car_a, frac2car_b, frac2car_c, temp_x, temp_y, temp_z);
    temp_frame_x = temp_x[0];
    temp_frame_y = temp_y[0];
    temp_frame_z = temp_z[0];
    distance = calc_dis(loc_x, loc_y, loc_z, temp_frame_x, temp_frame_y, temp_frame_z);

    return distance;
}



//calculate pure LJ(12-6) potentialat that point
double calc_pure_lj(double dis, double sigma1, double sigma2, double epsilon1, double epsilon2)
{
    double epsilon_c, sigma_c;
    double lj_potential = 0;

    sigma_c = (sigma1 + sigma2)/2;
    epsilon_c = sqrt(epsilon1*epsilon2);
    lj_potential = 4*epsilon_c*(pow((sigma_c/dis),12) - pow((sigma_c/dis),6));
    return lj_potential;
}



//calculate LJ potential with shift at that point
double calc_lj(double dis, double cutoff, double sigma, double sigma_frame_iiii, double epsilon, double epsilon_frame_iiii)
{
    double shift;
    double sigma_c;
    double lj_u = 0;
    if (dis <= cutoff)
    {
        sigma_c = (sigma+sigma_frame_iiii)/2;
        shift = calc_pure_lj(cutoff, sigma, sigma_frame_iiii, epsilon, epsilon_frame_iiii);
        if (dis < sigma_c*0.1)
        {
            dis = sigma_c*0.1;
        }
        lj_u = calc_pure_lj(dis, sigma, sigma_frame_iiii, epsilon, epsilon_frame_iiii) - shift;
    }
    return lj_u;
}



//calculate Feynman-Hibbs 4th order corrected LJ potentialat that point
double calc_pure_FH_lj(double dis, double sigma1, double sigma2, double epsilon1, double epsilon2, double mass1, double mass2,
                        double h_bar, double beta)
{
    double mu;                                                                  //reduced mass;
    double C1, C2;                                                              //abbreviated constant;
    double epsilon_c, sigma_c;
    double lj_potential = 0;

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

    sigma_c = (sigma1 + sigma2)/2;
    epsilon_c = sqrt(epsilon1*epsilon2);
    lj_potential = 4*epsilon_c* ( pow(sigma_c,12)/pow(dis,12) - pow(sigma_c,6)/pow(dis,6) ) +
                    C1*4*epsilon_c* ( 132*pow(sigma_c,12)/pow(dis,14) - 30*pow(sigma_c,6)/pow(dis,8) ) +
                    C2*4*epsilon_c* ( 23844*pow(sigma_c,12)/pow(dis,16) - 1590*pow(sigma_c,6)/pow(dis,10) );
    return lj_potential;
}



//calculate Feynman-Hibbs 4th order corrected LJ potential with shift at that point
double calc_FH_lj(double dis, double cutoff, double sigma, double sigma_frame_iiii, double epsilon, double epsilon_frame_iiii,
                    double mass1, double mass2, double h_bar, double beta)
{
    double shift;
    double sigma_c;
    double lj_u = 0;
    if (dis <= cutoff)
    {
        sigma_c = (sigma + sigma_frame_iiii)/2;
        shift = calc_pure_FH_lj(cutoff, sigma, sigma_frame_iiii, epsilon, epsilon_frame_iiii, mass1, mass2, h_bar, beta);
        if (dis < sigma_c*0.1)
        {
            dis = sigma_c*0.1;
        }
        lj_u = calc_pure_FH_lj(dis, sigma, sigma_frame_iiii, epsilon, epsilon_frame_iiii, mass1, mass2, h_bar, beta) - shift;
    }
    return lj_u;
}
