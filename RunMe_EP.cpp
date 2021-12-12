// #include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <sstream>
#include <time.h>
#include <cstring>

#define PI 3.141592653589793238
//Ns - rows, Ny - columns
#define Ns 150 
#define Ny 350
#define N_check 1 
#define Nt 10000000
#define Niter 10000

using namespace std;

//mean free path initialization
double l_mfp=6.0*pow(10,-9);			//5.7nm for NanoLetters	
double R=400.*pow(10,-9);			//cylinder radius in m	(280nm for NanoLetters)  *0.5
		
//kinetic coefficients definition

double c = 3.0 * pow(10, 8);		//light velocity in m/s
double vF = 6.0*pow(10, 5);			//Fermi velocity in m/s
double D = l_mfp*vF/3.0;			//diffusion coefficient in m^2/s
double kT = 0.770;					//dimensionless, equal T/Tc
double nV = 5.6*pow(10, 28);		//valency electrons concentration in 1/m^3
double C_e = 0.281*pow(10, -7);		//q^2_e/m_e in Si (Couloumb/kg)
double C_fi = 0.11*pow(10, -17);	//equal Plank constant divide 2x electron charge [erg/SGS]
double C_force = 9.08*pow(10, 20);	//dimensionless, equal (c*Hc*m_s)/(2*sqrt(2)*PI*e_s*h_planck) - coeff from LorForce 
double sigma_dim = l_mfp/(3.72*pow(10, -16)); //conductivity in 1/(Om*m)	//Drude model: C_e*pow(kT, 4)*l_mfp*nV / vF;	
//double lambda_0 = 39.0*pow(10, -9);	//const for lambda
//double xi_0 = 39.0*pow(10, -9);		//const for xi
double lambda_0 = 34.0*pow(10, -9);	//const for lambda
double xi_0 = 230.0*pow(10, -9);		//const for xi
double F_0 = 2.068*pow(10, -11);	//magnetic flow quant in Gauss*m^2

//dimensionless coefficients definition

double psi_0_square = 1.81*pow(10,20);								//psi^2_0 from equations
double lambda = lambda_0*sqrt(xi_0 / (2.0*(1.0 - kT)*1.33*l_mfp));	//London's penetration depth in m
double xi = 0.855*sqrt(xi_0*l_mfp / (1.0 - kT));					//coherency length in m   
double kappa = lambda / xi;											//GL parameter
double tau = pow(xi, 2) / D;										//characteristic time in s
double Hc = F_0 / (2.0*PI*lambda*xi*sqrt(2.0));						//magnetic quant unit in Gauss
double B_0 = sqrt(2.0)*Hc*pow(10, -4);								//magnetic field unit in Tesla
double fi_0 = pow(10,6)*300.0*kappa*D*C_fi/pow(xi,2);				//voltage unit in muV
double j_0 = (3.34*pow(10,-6))*F_0*c/(8.0*PI*PI*pow(lambda,2)*xi);	//current density unit
double sigma_0 = pow(c, 2)/(4.0*PI*kappa*kappa*D*8.988*pow(10,9));	//conductivity unit in 1/(Om*m)

//domain geometry parameters dimensionless

double delta = 60.0*pow(10, -9);	//cut width in m		(59nm for NanoLetters)   *0.5
double L = 5.*pow(10, -6);			//cylinder length in m	(3.5nm for NanoLetters)           *0.5
double R_nondim = R/lambda;         //dimensionless radius
double delta_nondim = delta/lambda; //dimensionless cut
double a = (2.*PI*R - delta)/lambda;//dimensionless S size	(a = 6.097 - length)
double b = L / lambda;				//dimensionless Y size	(b = 12.544 - width)

//process parameters initialization part 2

double sigma = sigma_dim / sigma_0;		//conductivity dimensionless
double B_dim, B_ind;					//magnetic field	
double j_tr_dim, j_tr, j_S;				//transport current
double free_energy_global; 				//free energy 

//numerical domain steps
double hs = a/Ns;
double hy = b/Ny;
double h = max(hs,hy); 
double ht = 1.0*h*h;

//scalar potential solver parameters
double h_tau = ht/10.; 
double eps = 0.00002;			
double mas_error;	

int N= max(Ns, Ny);
int time_count, ev_count;				 

double poisonns_left, poisonns_right;

//functions initialization
double ReDss(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUs_mid, double ImUs_mid, double ReUs_prev, double ImUs_prev, double hs);
double ImDss(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUs_mid, double ImUs_mid, double ReUs_prev, double ImUs_prev, double hs);
double ReDyy(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUy_mid, double ImUy_mid, double ReUy_prev, double ImUy_prev, double hy);
double ImDyy(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUy_mid, double ImUy_mid, double ReUy_prev, double ImUy_prev, double hy);
double As_value(double s, double y, double B_ind);
double Ay_value(double s, double y, double B_ind, double a);
double mod_PSI(double psi_RE, double psi_IM); 
double max_mass(double **mas, int count_s, int count_y);
void vortex_catch(double time_count, double **psi1, double **psi2, double *y);

int main(int argc, char *argv[])
{
	srand(time(NULL));
/*
    //declaration of auxiliary variable, ID and number of processes
    int rc, myid, np;
    //initialization of programm
    rc = MPI_Init(&argc,&argv);
    if(rc != MPI_SUCCESS) 
    {
        cout << "Error starting MPI programm. Terminating." << endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    //definition of process ID and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &np); 
*/
//output filename
char *Magnetic;
Magnetic = new char[50];

// if(myid==0)
// {
strcpy(Magnetic, "B\0");

char *coeff_out;
coeff_out = new char[40];
strcpy(coeff_out, "coefficients_\0");
strcat(coeff_out, Magnetic);
strcat(coeff_out, ".txt\0"); 

ofstream out(coeff_out,ios::app);  

B_dim = 10.0;							//magnetic field in mT  *4.0
B_ind = 0.001*B_dim / B_0;				//magnetic field dimensionless
 
j_tr_dim = ((1.0e-9)+0.877)*20.0*pow(10, 9);
j_tr = j_tr_dim / j_0; 				//transport current dimensionless   *8.0
j_S = j_tr_dim*L*50.0*pow(10,-9);
 
 // Ð Ñ—Ð Â°Ð¡Ð‚Ð Â°Ð Ñ˜Ð ÂµÐ¡â€šÐ¡Ð‚Ð¡â€¹ Ð¡Ð‚Ð Â°Ð¡ÐƒÐ¡â€¡Ð ÂµÐ¡â€šÐ Â°
 
out << "Xi(m) = " << xi << " Lambda(m) = " << lambda << " Hc = " << Hc << " B_0 = " << B_0 <<endl;
out << "Sigma = " << sigma_dim << " B(T) = " << B_dim << " j_tr(A/m^2) = " << j_tr_dim  << " sigma_nondim = " << sigma << " j_tr_nondim " << j_tr << endl;
out << "R(m) = " << R <<" L(m) = " << L << " mean free path(m) = " << l_mfp << " GL parameter = " << kappa << endl;
out << "R_nondim = " << R_nondim << " delta_nondim = " << delta_nondim << " a_nondim " << a << " L_nondim = " << b << endl;
out << "Ns = " << Ns << " Ny = " << Ny << " Ncheck = " << N_check << " hs = " << hs << " hy = " << hy << " ht = " << ht << " Nt = " << Nt << " h_tau = "<< h_tau;

out.close();

 
// }

	//for voltage analysis
	int volt_count=0;		
	double fi_right[N_check], fi_left[N_check], av_voltage[N_check];
	double voltage=0.;
	for (int i = 0; i < N_check; i++)
	{
		fi_right[i] = 0.;
		fi_left[i] = 0.;
		av_voltage[i] = 0.;
	}
	
	//initialize 1D dynamic mesh for s and y coord dimensionless
	double *s = new double[Ns + 1];
	double *y = new double[Ny + 1];

	//initialize 1D dynamic mesh for s and y coord dimensions
	double *s_dim = new double[Ns + 1];
	double *y_dim = new double[Ny + 1];

	//initialize 2D dynamic mesh for vector-potential
	double **kappa_rand = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		kappa_rand[i] = new double[Ny + 1];
	
	//initialize 2D dynamic mesh for vector-potential
	double **As = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		As[i] = new double[Ny + 1];

	double **Ay = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		Ay[i] = new double[Ny + 1];

	//initialize 2D dynamic mesh for link-variables
	double **ReUs = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		ReUs[i] = new double[Ny + 1];

	double **ImUs = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		ImUs[i] = new double[Ny + 1];

	double **ReUy = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		ReUy[i] = new double[Ny + 1];

	double **ImUy = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		ImUy[i] = new double[Ny + 1];

	//initialize 2D dynamic mesh for initial condition
	double **psi_start1 = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi_start1[i] = new double[Ny + 1]; //Real part

	double **psi_start2 = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi_start2[i] = new double[Ny + 1]; //Imaginary part

	//initialize 2D dynamic mesh for calculating
	double **psi1 = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi1[i] = new double[Ny + 1]; //Real part

	double **psi2 = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi2[i] = new double[Ny + 1]; //Imaginary part
		
	//initialize additional 2D dynamic mesh for time updating 
	double **psi1_old = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi1_old[i] = new double[Ny + 1]; //Real part

	double **psi2_old = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		psi2_old[i] = new double[Ny + 1]; //Imaginary part

	//initialize 2D dynamic mesh for scalar-potential
	double **fi = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		fi[i] = new double[Ny + 1];

	double **fi_half = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		fi_half[i] = new double[Ny + 1];

	double **fi_ev = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		fi_ev[i] = new double[Ny + 1];
		
	double **fi_old = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		fi_old[i] = new double[Ny + 1];
	
	//initialize 2D dynamic mesh for scalar-potential
	double **j_sc_y = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_sc_y[i] = new double[Ny + 1];

	double **j_sc_s = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_sc_s[i] = new double[Ny + 1];
	
	double **j_norm_y = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_norm_y[i] = new double[Ny + 1];

	double **j_norm_s = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_norm_s[i] = new double[Ny + 1];
	
	double **j_tot_y = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_tot_y[i] = new double[Ny + 1];

	double **j_tot_s = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		j_tot_s[i] = new double[Ny + 1];
		
	double **div_j = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		div_j[i] = new double[Ny + 1];
		
	double **free_energy = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		free_energy[i] = new double[Ny + 1];	
	
	//initialize 2D dynamic mesh for Lorentz force
	double **F_fi = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		F_fi[i] = new double[Ny + 1];	
	
	double **F_y = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		F_y[i] = new double[Ny + 1];
		
	double **B_n = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		B_n[i] = new double[Ny + 1];
		
	double **error = new double*[Ns + 1];
	for (int i = 0; i < Ns + 1; i++)
		error[i] = new double[Ny + 1];

	//total cleaning mesh for calculations
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			psi1[i][j] = 0.0;
			psi2[i][j] = 0.0;
			fi[i][j] = 0.0;
			fi_half[i][j] = 0.0;
			fi_ev[i][j] = 0.0;
			fi_old[i][j] = 0.0;
			F_fi[i][j] = 0.0;
			j_sc_y[i][j] = 0.0;
			j_sc_s[i][j] = 0.0;
			j_norm_y[i][j] = 0.0;
			j_norm_s[i][j] = 0.0;
			j_tot_y[i][j] = 0.0;
			j_tot_s[i][j] = 0.0;
			free_energy[i][j] = 0.0;
			div_j[i][j] = 0.0;
			F_y[i][j] = 0.0;
			error[i][j] = 0.0;
		}

	//generate random initial conditions from 0 to 1
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			psi_start1[i][j] = 0.01*(rand() % 100);
			psi_start2[i][j] = 0.01*(rand() % 100);
		}

	//norming on sqrt(2) to control |psi_start|^2 < 1
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			psi_start1[i][j] = (1.0 / 1.414)*psi_start1[i][j];
			psi_start2[i][j] = (1.0 / 1.414)*psi_start2[i][j];
		}

	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			psi1[i][j] = psi_start1[i][j];
			psi2[i][j] = psi_start2[i][j];
		}

	//filling S mesh with coord values
	for (int i = 0; i < Ns + 1; i++)
	{
		s[i] = i*hs;
	}

	for (int i = 0; i < Ns + 1; i++)
	{
		s_dim[i] = pow(10,9)*(i*hs*lambda + delta/2.0);		//in nm
	}

	//filling Y mesh with coord values
	for (int j = 0; j < Ny + 1; j++)
	{
		y[j] = j*hy;
	}

	for (int j = 0; j < Ny + 1; j++)
	{
		y_dim[j] = pow(10, 9)*j*hy*lambda;					//in nm
	}

	//inhomogeneous kappa-----------------------------------------------------------------------------BEGIN
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			kappa_rand[i][j] = kappa; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------1--------------------------------------------
	for (int i = 10; i < 10+11; i++) //150
		for (int j = 50; j < 50+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------2--------------------------------------------	
	for (int i = 19; i < 19+11; i++) //150
		for (int j = 250; j < 250+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------3--------------------------------------------
	for (int i = 100; i < 100+11; i++) //150
		for (int j = 95; j < 95+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------4--------------------------------------------	
	for (int i = 120; i < 120+11; i++) //150
		for (int j = 300; j < 300+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------5--------------------------------------------	
	for (int i = 40; i < 40+11; i++) //150
		for (int j = 210; j < 210+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------6--------------------------------------------	
	for (int i = 30; i < 30+11; i++) //150
		for (int j = 160; j < 160+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------7--------------------------------------------	
	for (int i = 90; i < 90+11; i++) //150
		for (int j = 10; j < 10+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------8--------------------------------------------	
	for (int i = 35; i < 35+11; i++) //150
		for (int j = 180; j < 180+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------9--------------------------------------------	
	for (int i = 25; i < 25+11; i++) //150
		for (int j = 320; j < 320+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------10--------------------------------------------	
	for (int i = 65; i < 65+11; i++) //150
		for (int j = 70; j < 70+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------11--------------------------------------------	
	for (int i = 10; i < 10+11; i++) //150
		for (int j = 280; j < 280+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------12--------------------------------------------	
	for (int i = 130; i < 130+11; i++) //150
		for (int j = 200; j < 200+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------13--------------------------------------------	
	for (int i = 80; i < 80+11; i++) //150
		for (int j = 90; j < 90+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------14--------------------------------------------	
	for (int i = 76; i < 76+11; i++) //150
		for (int j = 180; j < 180+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}
	//----------------------15--------------------------------------------	
	for (int i = 10; i < 10+11; i++) //150
		for (int j = 250; j < 250+11; j++) //350
		{
			kappa_rand[i][j] = kappa*0.9; // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.0*( (rand() % 100 - 49.5) / 49.5 ) ); // 10% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.1*( (rand() % 100 - 49.5) / 49.5 ) ); // 20% scattering
			//kappa_rand[i][j] = kappa*(1.0 + 0.15*( (rand() % 100 - 49.5) / 49.5 ) ); // 30% scattering
		}		
	//inhomogeneous kappa-----------------------------------------------------------------------------END
	
	//vector-potential values
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			As[i][j] = As_value(s[i] + hs / 2.0, y[j], B_ind);
			Ay[i][j] = Ay_value(s[i], y[j] + hy / 2.0, B_ind, a);
		}

	//link-variables values
	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			ReUs[i][j] = cos(hs*kappa_rand[i][j]*As[i][j]);
			ImUs[i][j] = sin(hs*kappa_rand[i][j]*As[i][j]); //**** Dolzhen li zdes' byt' minus
			ReUy[i][j] = cos(hy*kappa_rand[i][j]*Ay[i][j]); 
			ImUy[i][j] = sin(hy*kappa_rand[i][j]*Ay[i][j]); //**** Dolzhen li zdes' byt' minus
		}

	for (int i = 0; i < Ns + 1; i++)
		for (int j = 0; j < Ny + 1; j++)
		{
			B_n[i][j] = B_dim*sin(s[i]*lambda/R);
		}
	
	//saving values of normal to surface component B_n to file
	char *norm_comp;
	norm_comp = new char[40];
	strcpy(norm_comp, "norm_comp_\0");
	strcat(norm_comp, Magnetic);
	strcat(norm_comp, ".txt\0");
	
	ofstream out_bn(norm_comp, ios::app);

	for (int i = 0; i < Ns + 1; i++)
		{
		for (int j = 0; j < Ny + 1; j++)
			{
			out_bn <<  B_n[i][j] << '\t';
           	if(j != Ny)
             	out_bn << "  ";
           	else 
            	out_bn << endl; 
				}
			}		
	out_bn.close();
		
	time_count = 0;
	ev_count = 0;
	double time = 0.0;
		
	//for time step updating
	int var_err_1 = 0;
	double error_var, error_crit, var_temp;
	
	do
	{
		
		if(time_count <= (Niter-1))
		{
		
		//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
		for (int i = 1; i < Ns; i++)
		{
			for (int j = 1; j < Ny; j++)
			{
				psi1[i][j] = psi_start1[i][j] + ht*(
				(ReDss(psi_start1[i + 1][j], psi_start2[i + 1][j], psi_start1[i][j], psi_start1[i - 1][j], psi_start2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
				ReDyy(psi_start1[i][j + 1], psi_start2[i][j + 1], psi_start1[i][j], psi_start1[i][j - 1], psi_start2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j-1], ImUy[i][j-1], hy))/(kappa_rand[i][j]*kappa_rand[i][j]) + 
				psi_start1[i][j] * (1.0 - mod_PSI(psi_start1[i][j], psi_start2[i][j])) + kappa_rand[i][j]*fi[i][j] * psi_start2[i][j]);
				psi2[i][j] = psi_start2[i][j] + ht*(
				(ImDss(psi_start1[i + 1][j], psi_start2[i + 1][j], psi_start2[i][j], psi_start1[i - 1][j], psi_start2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
				ImDyy(psi_start1[i][j + 1], psi_start2[i][j + 1], psi_start2[i][j], psi_start1[i][j - 1], psi_start2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j - 1], ImUy[i][j - 1], hy)) / (kappa_rand[i][j]*kappa_rand[i][j]) + 
				psi_start2[i][j] * (1.0 - mod_PSI(psi_start1[i][j], psi_start2[i][j])) - kappa_rand[i][j]*fi[i][j] * psi_start1[i][j]);
			}
		}

		//neumann b.c.s at s=0 and s=a
		for (int j = 0; j < Ny+1; j++)
		{
			/*
			psi1[0][j] = psi1[1][j] * ReUs[0][j] + psi2[1][j] * ImUs[0][j];
			psi2[0][j] = psi2[1][j] * ReUs[0][j] - psi1[1][j] * ImUs[0][j];

			psi1[Ns][j] = psi1[Ns - 1][j] * ReUs[Ns - 1][j] - psi2[Ns - 1][j] * ImUs[Ns - 1][j];
			psi2[Ns][j] = psi2[Ns - 1][j] * ReUs[Ns - 1][j] + psi1[Ns - 1][j] * ImUs[Ns - 1][j];
			*/
			
			psi1[0][j] = 0.0;
			psi2[0][j] = 0.0;

			psi1[Ns][j] = 0.0;
			psi2[Ns][j] = 0.0;
			
		}

		//neumann b.c.s at y=0 and y=b
		for (int i = 1; i < Ns; i++)//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
		{
			psi1[i][0] = psi1[i][1] * ReUy[i][0] + psi2[i][1] * ImUy[i][0];
			psi2[i][0] = psi2[i][1] * ReUy[i][0] - psi1[i][1] * ImUy[i][0];

			psi1[i][Ny] = psi1[i][Ny - 1] * ReUy[i][Ny - 1] - psi2[i][Ny - 1] * ImUy[i][Ny - 1];
			psi2[i][Ny] = psi2[i][Ny - 1] * ReUy[i][Ny - 1] + psi1[i][Ny - 1] * ImUy[i][Ny - 1];
		}
		
		for (int i = 0; i < Ns + 1; i++)
			for (int j = 0; j < Ny + 1; j++)
			{
				psi_start1[i][j] = psi1[i][j];
				psi_start2[i][j] = psi2[i][j];
			}
			
		}
		
		if(time_count > (Niter - 1))
		{
		//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
		for (int i = 1; i < Ns; i++)
		{
			for (int j = 1; j < Ny; j++)
			{
				psi1[i][j] = psi_start1[i][j] + ht*(
				(ReDss(psi_start1[i + 1][j], psi_start2[i + 1][j], psi_start1[i][j], psi_start1[i - 1][j], psi_start2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
				ReDyy(psi_start1[i][j + 1], psi_start2[i][j + 1], psi_start1[i][j], psi_start1[i][j - 1], psi_start2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j-1], ImUy[i][j-1], hy))/(kappa_rand[i][j]*kappa_rand[i][j]) + 
				psi_start1[i][j] * (1.0 - mod_PSI(psi_start1[i][j], psi_start2[i][j])) + kappa_rand[i][j]*fi[i][j] * psi_start2[i][j]);
				psi2[i][j] = psi_start2[i][j] + ht*(
				(ImDss(psi_start1[i + 1][j], psi_start2[i + 1][j], psi_start2[i][j], psi_start1[i - 1][j], psi_start2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
				ImDyy(psi_start1[i][j + 1], psi_start2[i][j + 1], psi_start2[i][j], psi_start1[i][j - 1], psi_start2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j - 1], ImUy[i][j - 1], hy)) / (kappa_rand[i][j]*kappa_rand[i][j]) + 
				psi_start2[i][j] * (1.0 - mod_PSI(psi_start1[i][j], psi_start2[i][j])) - kappa_rand[i][j]*fi[i][j] * psi_start1[i][j]);
			}
		}

		//neumann b.c.s at s=0 and s=a
		for (int j = 0; j < Ny+1; j++)
		{
			/*
			psi1[0][j] = psi1[1][j] * ReUs[0][j] + psi2[1][j] * ImUs[0][j];
			psi2[0][j] = psi2[1][j] * ReUs[0][j] - psi1[1][j] * ImUs[0][j];

			psi1[Ns][j] = psi1[Ns - 1][j] * ReUs[Ns - 1][j] - psi2[Ns - 1][j] * ImUs[Ns - 1][j];
			psi2[Ns][j] = psi2[Ns - 1][j] * ReUs[Ns - 1][j] + psi1[Ns - 1][j] * ImUs[Ns - 1][j];
			*/
			
			psi1[0][j] = 0.0;
			psi2[0][j] = 0.0;

			psi1[Ns][j] = 0.0;
			psi2[Ns][j] = 0.0;
			
		}

		//neumann b.c.s at y=0 and y=b
		for (int i = 1; i < Ns; i++)//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
		{
			psi1[i][0] = psi1[i][1] * ReUy[i][0] + psi2[i][1] * ImUy[i][0];
			psi2[i][0] = psi2[i][1] * ReUy[i][0] - psi1[i][1] * ImUy[i][0];

			psi1[i][Ny] = psi1[i][Ny - 1] * ReUy[i][Ny - 1] - psi2[i][Ny - 1] * ImUy[i][Ny - 1];
			psi2[i][Ny] = psi2[i][Ny - 1] * ReUy[i][Ny - 1] + psi1[i][Ny - 1] * ImUy[i][Ny - 1];
		}
		/*
		for (int i = 0; i < Ns + 1; i++)
			for (int j = 0; j < Ny + 1; j++)
			{
				psi_start1[i][j] = psi1[i][j];
				psi_start2[i][j] = psi2[i][j];
			}
		*/	
		
		for (int i = 0; i < Ns + 1; i++)
			for (int j = 0; j < Ny + 1; j++)
			{
				psi_start1[i][j] = psi1[i][j];
				psi_start2[i][j] = psi2[i][j];
			}  	

		if((time_count%100)==0)
		{
		
    	 	//vortex_catch(time_count, psi1, psi2, y);

		for (int i = 0; i < Ns + 1; i++)
			for (int j = 0; j < Ny + 1; j++)
			{
				fi[i][j]=fi_old[i][j];//fi[i][j] = 0.0;
			}

		do{

			//classic fully explicit scheme
			for (int i = 1; i < Ns; i++)
				for (int j = 1; j < Ny; j++)//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
				{
					poisonns_left = (fi[i + 1][j] - 2.0*fi[i][j] + fi[i - 1][j]) / (hs*hs) + (fi[i][j + 1] - 2.0*fi[i][j] + fi[i][j - 1]) / (hy*hy);
					poisonns_right = (1.0 / (kappa_rand[i][j]*sigma))*(psi1[i][j] * (
					ImDss(psi1[i + 1][j], psi2[i + 1][j], psi2[i][j], psi1[i - 1][j], psi2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
					ImDyy(psi1[i][j + 1], psi2[i][j + 1], psi2[i][j], psi1[i][j - 1], psi2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j-1], ImUy[i][j-1], hy)) - 
					psi2[i][j] * (
					ReDss(psi1[i + 1][j], psi2[i + 1][j], psi1[i][j], psi1[i - 1][j], psi2[i - 1][j], ReUs[i][j], ImUs[i][j], ReUs[i - 1][j], ImUs[i - 1][j], hs) + 
					ReDyy(psi1[i][j + 1], psi2[i][j + 1], psi1[i][j], psi1[i][j - 1], psi2[i][j - 1], ReUy[i][j], ImUy[i][j], ReUy[i][j-1], ImUy[i][j-1], hy)));
					fi_ev[i][j] = fi[i][j] + h_tau*(poisonns_left - poisonns_right);
				}

			//neumann b.c.s at y=0 and y=b
			for (int i = 0; i < Ns + 1; i++)
			{
				fi_ev[i][0] = fi_ev[i][1];
				fi_ev[i][Ny] = fi_ev[i][Ny - 1];
			}

			//neumann b.c.s at s=0 and s=a
			for (int j = 0; j < Ny + 1; j++)
			{
				fi_ev[0][j] = fi_ev[1][j] + 1.*hs*j_tr / sigma;
				fi_ev[Ns][j] = fi_ev[Ns - 1][j] - 1.*hs*j_tr / sigma;
			}

			//evolution error estimating
			for (int i = 0; i < Ns + 1; i++)
				for (int j = 0; j < Ny + 1; j++)
				{
					error[i][j] = fabs(fi_ev[i][j] - fi[i][j]);
				}
			mas_error = max_mass(error, Ns + 1, Ny + 1);

			for (int i = 0; i < Ns + 1; i++)
				for (int j = 0; j < Ny + 1; j++)
				{
					fi[i][j] = fi_ev[i][j];
				}

			ev_count++;

		} while (mas_error > eps);
		
			for (int i = 0; i < Ns + 1; i++)
				for (int j = 0; j < Ny + 1; j++)
				{
					fi_old[i][j] = fi[i][j];
				}
		
		
		}
		
		}
		
		if(time_count % 500 == 0) 
		{

			for (int i = 0; i < Ns + 1; i++)
			{ 
				F_fi[i][0] = 0.;
				F_fi[i][Ny] = 0.;
			}
			
			//maybe change the sign j_tr
			for (int j = 0; j < Ny + 1; j++)
			{
				F_y[0][j] = j_tr*B_ind;
				F_y[Ns][j] = (-1.)*j_tr*B_ind;
			}

			for (int i = 0; i < Ns + 1; i++)//**** Proverit' eti uravneniya; osobenno s uchetom strochek 368, 370;
				for (int j = 0; j < Ny; j++)
				{
					j_sc_y[i][j] = (1. / (kappa_rand[i][j]*hy))*(ReUy[i][j] * (psi2[i][j + 1] * psi1[i][j] - psi1[i][j + 1] * psi2[i][j]) - ImUy[i][j] * (psi1[i][j + 1] * psi1[i][j] + psi2[i][j + 1] * psi2[i][j]));
					// F_fi[i][j] = (-1.)*(j_sc_y[i][j] - (C_force/psi_0_square)*(sigma/kappa)*(fi[i][j+1]-fi[i][j])/hy)*B_ind*sin(s[i]*lambda/R);
					j_norm_y[i][j] = (-1.0)*sigma*(fi[i][j+1]-fi[i][j])/(1.0*hy); 
					j_tot_y[i][j] = j_sc_y[i][j] + j_norm_y[i][j];
				}
			
			for (int i = 0; i < Ns; i++)
				for (int j = 0; j < Ny+1; j++)
				{
					j_sc_s[i][j] = (1. / (kappa_rand[i][j]*hs))*(ReUs[i][j] * (psi2[i+1][j] * psi1[i][j] - psi1[i+1][j] * psi2[i][j]) - ImUs[i][j] * (psi1[i+1][j] * psi1[i][j] + psi2[i+1][j] * psi2[i][j]));	// îòêóäà ìíîæèòåëü *(lambda/R) ïðè âûâîäå ó Ïîñåíèöêîãî		
					// F_y[i][j] = (-1.)*(j_sc_s[i][j]*lambda/R + (C_force/psi_0_square)*(sigma/kappa)*(fi[i+1][j]-fi[i][j])/hs)*B_ind*sin(s[i]*lambda/R);
					j_norm_s[i][j] = (-1.)*sigma*(fi[i+1][j]-fi[i][j])/(1.0*hs);
					j_tot_s[i][j] = j_sc_s[i][j] + j_norm_s[i][j];	
				}
			
			for (int i = 1; i < Ns; i++)
				for (int j = 1; j < Ny; j++)
					div_j[i][j] = (j_tot_s[i][j] - j_tot_s[i-1][j])/(1.0*hs) + (j_tot_y[i][j] - j_tot_y[i][j-1])/(1.0*hy);
			
			//free energy calculation
			for (int i = 1; i < Ns; i++)
				for (int j = 1; j < Ny; j++)
					free_energy[i][j] = -mod_PSI(psi1[i][j], psi2[i][j]) + mod_PSI(psi1[i][j], psi2[i][j])*mod_PSI(psi1[i][j], psi2[i][j])*0.5 + 
									(1.0/kappa_rand[i][j]*kappa_rand[i][j])*((mod_PSI(ReUs[i][j]*psi1[i+1][j] + ImUs[i][j]*psi2[i+1][j] - psi1[i][j], ReUs[i][j]*psi2[i+1][j] - ImUs[i][j]*psi1[i+1][j] - psi2[i][j]))/(hs*hs) + 
														(mod_PSI(ReUy[i][j]*psi1[i][j+1] + ImUy[i][j]*psi2[i][j+1] - psi1[i][j], ReUy[i][j]*psi2[i][j+1] - ImUy[i][j]*psi1[i][j+1] - psi2[i][j]))/(hy*hy));
			free_energy_global = 0.0;
			for (int i = 1; i < Ns; i++)
				for (int j = 1; j < Ny; j++)
					free_energy_global = free_energy_global + free_energy[i][j];
			
			free_energy_global = free_energy_global*hs*hy;
					
		}
		//cout << ev_count << endl;
		ev_count = 0;

		time_count++;
		time = time + ht;
				
		//voltage analysis
		char *out_voltage;
		out_voltage = new char[50];

		strcpy(out_voltage, "delta_FI_\0");
     	strcat(out_voltage, Magnetic);
        strcat(out_voltage, ".txt\0"); 
	    ofstream out(out_voltage,ios::app);  
	  	if(!out)
      	cerr<<"Error at the opening!!!.\n"<<endl;
		out << time*tau*pow(10,9) << " ";///1000 << " ";//
		for (int i = 0; i < N_check; i++)
		{
		
				for (int j = 0; j < Ny; j++)
				{
					fi_right[i] += fi[Ns - i][j];
					fi_left[i] += fi[i][j];
				}
			av_voltage[i] = (fi_right[i] - fi_left[i]) / Ny;
			out << av_voltage[i]*fi_0 << " ";	//add /j_S to av_voltage[i] to define R(t)
			
			
			if((time_count>50000) && (i==N_check-1))
			{
				voltage += av_voltage[i];
				volt_count++;
			}
			
			fi_right[i] = 0.;
			fi_left[i] = 0.;
			av_voltage[i] = 0.;
		}
		out << endl;
		if(time_count==Nt)
		{
			out<<"Average voltage between  = "<< voltage*fi_0/volt_count << " muV" << endl;
		}
		out.close();
		
		//energy analysis
		if ((time_count >= 8000) && (time_count % 500 == 0)){
		
			char *out_global_energy;
			out_global_energy = new char[50];

			strcpy(out_global_energy, "global_energy_\0");
     		strcat(out_global_energy, Magnetic);
        	strcat(out_global_energy, ".txt\0"); 
	    	ofstream out(out_global_energy,ios::app);  
	  		if(!out)
      			cerr<<"Error at the opening!!!.\n"<<endl;
			out << time*tau*pow(10,9) << " ";///1000 << " ";//
			out << free_energy_global << " ";	//add /j_S to av_voltage[i] to define R(t)
			out << endl;		
			out.close();
		}

		if ((time_count >= 8000) && (time_count % 10000 == 0))
		{
	  		
		 
			char *out_parameter_order;
			char *kappa_rand_file;
	  		char *absolute_value;
	  		char *out_scalar_potential;
	  		char *lorentz_force_fi;
	  	    char *lorentz_force_y;
	  	    char *current_fi;
	  	    char *current_y;	
			char *current_fi_norm;
	  	    char *current_y_norm;	  	    
			char *current_fi_tot;
	  	    char *current_y_tot;
			char *div_current;    	    
			char *free_energy_out;
	  	    
      		out_parameter_order = new char[50];
			kappa_rand_file = new char[50];
      		absolute_value = new char[50];
      		out_scalar_potential = new char[50];
    	  	lorentz_force_fi = new char[50] ;
	  	    lorentz_force_y = new char[50];
	  	    current_fi = new char[50];
	  	    current_y = new char[50];
	  	    current_fi_norm = new char[50];
	  	    current_y_norm = new char[50];
	  	    current_fi_tot = new char[50];
	  	    current_y_tot = new char[50];
	  	    div_current = new char[50];
	  	    free_energy_out = new char[50];
	  	    
	  	    char tt[8];
      		//definition of process number in char-type; for the file-name
      		int i_1000000, i_100000, i_10000, i_1000, i_100, i_10, i_1;
      		i_1000000 = time_count/1000000;
      		i_100000 = (time_count-1000000*i_1000000)/100000;
      		i_10000 = (time_count-1000000*i_1000000-100000*i_100000)/10000;
      		i_1000 = (time_count-1000000*i_1000000-100000*i_100000-10000*i_10000)/1000;
      		i_100 = (time_count-1000000*i_1000000-100000*i_100000-10000*i_10000-1000*i_1000)/100;
      		i_10  = (time_count-1000000*i_1000000-100000*i_100000-10000*i_10000-1000*i_1000-100*i_100)/10;
      		i_1   = (time_count-1000000*i_1000000-100000*i_100000-10000*i_10000-1000*i_1000-100*i_100-10*i_10);
      		tt[0] = i_1000000 + 48;
      		tt[1] = i_100000  + 48;
      		tt[2] = i_10000   + 48;
      		tt[3] = i_1000    + 48;
      		tt[4] = i_100     + 48;
      		tt[5] = i_10      + 48;
      		tt[6] = i_1       + 48;
      		tt[7] = '\0';

//-----------------0---------------------------------------------------    
       		strcpy(kappa_rand_file, "kappa_\0");
      		strcat(kappa_rand_file, tt);
      		strcat(kappa_rand_file, ".txt\0"); 
	  		ofstream out0(kappa_rand_file,ios::app);  
	  		if(!out0)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out0 << time*tau*pow(10,9) << "  " << endl; 
			
	  		for (int i = 0; i < Ns + 1; i++)
			{
				for (int j = 0; j < Ny + 1; j++)
				{
				out0 <<  kappa_rand[i][j] << '\t';
           			if(j != Ny)
             			out0 << "  ";
           			else 
            			out0 << endl; //data1 << i*hs*lambda + delta / 2. << " " << j*hy*lambda << " " << mod_PSI(psi1[i][j], psi2[i][j]) << endl;
				}
			}
			
			out0 << endl << endl;
			out0.close();
			delete []kappa_rand_file;
//-----------------1---------------------------------------------------    
       		strcpy(absolute_value, "mod_PSI_\0");
      		strcat(absolute_value, tt);
      		strcat(absolute_value, ".txt\0"); 
	  		ofstream out1(absolute_value,ios::app);  
	  		if(!out1)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out1 << time*tau*pow(10,9) << "  " << endl; 
			
	  		for (int i = 0; i < Ns + 1; i++)
			{
				for (int j = 0; j < Ny + 1; j++)
				{
				out1 <<  mod_PSI(psi1[i][j], psi2[i][j]) << '\t';
           			if(j != Ny)
             			out1 << "  ";
           			else 
            			out1 << endl; //data1 << i*hs*lambda + delta / 2. << " " << j*hy*lambda << " " << mod_PSI(psi1[i][j], psi2[i][j]) << endl;
				}
			}
			
			out1 << endl << endl;
			out1.close();
			delete []absolute_value;
//------------------------------------------------2------------------------------------------------
			strcpy(out_parameter_order, "PSI_\0");
      		strcat(out_parameter_order, tt);
      		strcat(out_parameter_order, ".txt\0"); 
	  		/*ofstream out2(out_parameter_order,ios::app);  
	  		if(!out2)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out2 << time*tau*pow(10,9) << "  " << endl; 
			
	  		for (int i = 0; i < Ns + 1; i++)
			{
				for (int j = 0; j < Ny + 1; j++)
				{
				out2 <<  psi1[i][j] << " " << psi2[i][j] << '\t';
           			if(j != Ny)
             			out1 << "  ";
           			else 
            			out2 << endl; //data1 << i*hs*lambda + delta / 2. << " " << j*hy*lambda << " " << mod_PSI(psi1[i][j], psi2[i][j]) << endl;
				}
			}
			
			out2 << endl << endl;
			out2.close();*/
			delete []out_parameter_order;
//-------------------------------3--------------------------------------------------------			
			strcpy(out_scalar_potential, "FI_\0");
      		strcat(out_scalar_potential, tt);
      		strcat(out_scalar_potential, ".txt\0"); 
	  		/*ofstream out3(out_scalar_potential,ios::app);  
	  		if(!out3)
      		cerr<<"Error at the opening!!!.\n"<<endl;
      		
      		out3 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns + 1 ; i++) 
			{
				for (int j = 0; j < Ny + 1; j++)
				{
				out3 <<  fi[i][j] * fi_0 << '\t';
           			if(j != Ny)
             			out3 << "  ";
           			else 
            			out3 << endl;
				}
			}
			out3 << endl << endl;
			out3.close();*/
			delete []out_scalar_potential;
//---------------------------------6---------------------------------------------------			
			strcpy(current_fi, "current_fi\0");
			strcat(current_fi, tt);
			strcat(current_fi, ".txt\0");
			/*ofstream out6(current_fi, ios::app);
			if(!out6)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out6 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out6 << j_sc_s[i][j] << '\t';
           			if(j != Ny)
             			out6 << "  ";
           			else 
            			out6 << endl;
				}
			}
			out6 << endl << endl;
			out6.close();*/
			delete []current_fi;
//--------------------------7----------------------------------------------
			strcpy(current_y, "current_y\0");
			strcat(current_y, tt);
			strcat(current_y, ".txt\0");
			/*ofstream out7(current_y, ios::app);
			if(!out7)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out7 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out7 << j_sc_y[i][j] << '\t';
           			if(j != Ny)
             			out7 << "  ";
           			else 
            			out7 << endl;
				}
			}
			out7 << endl << endl;
			out7.close();*/
			delete []current_y;
//---------------------------8-----------------------------------------------
			strcpy(current_fi_norm, "current_fi_norm\0");
			strcat(current_fi_norm, tt);
			strcat(current_fi_norm, ".txt\0");
			/*ofstream out8(current_fi_norm, ios::app);
			if(!out8)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out8 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out8 << j_norm_s[i][j] << '\t';
           			if(j != Ny)
             			out8 << "  ";
           			else 
            			out8 << endl;
				}
			}
			out8 << endl << endl;
			out8.close();*/
			delete []current_fi_norm;
//--------------------------9----------------------------------------------
			strcpy(current_y_norm, "current_y_norm\0");
			strcat(current_y_norm, tt);
			strcat(current_y_norm, ".txt\0");
			/*ofstream out9(current_y_norm, ios::app);
			if(!out9)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out9 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out9 << j_norm_y[i][j] << '\t';
           			if(j != Ny)
             			out9 << "  ";
           			else 
            			out9 << endl;
				}
			}
			out9 << endl << endl;
			out9.close();*/
			delete []current_y_norm;
//---------------------------10-----------------------------------------------
			strcpy(current_fi_tot, "current_fi_tot\0");
			strcat(current_fi_tot, tt);
			strcat(current_fi_tot, ".txt\0");
			/*ofstream out10(current_fi_tot, ios::app);
			if(!out10)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out10 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out10 << j_tot_s[i][j] << '\t';
           			if(j != Ny)
             			out10 << "  ";
           			else 
            			out10 << endl;
				}
			}
			out10 << endl << endl;
			out10.close();*/
			delete []current_fi_tot;
//--------------------------11----------------------------------------------
			strcpy(current_y_tot, "current_y_tot\0");
			strcat(current_y_tot, tt);
			strcat(current_y_tot, ".txt\0");
			/*ofstream out11(current_y_tot, ios::app);
			if(!out11)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out11 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out11 << j_tot_y[i][j] << '\t';
           			if(j != Ny)
             			out11 << "  ";
           			else 
            			out11 << endl;
				}
			}
			out11 << endl << endl;
			out11.close();*/
			delete []current_y_tot;
//--------------------------12----------------------------------------------
			strcpy(free_energy_out, "free_energy\0");
			strcat(free_energy_out, tt);
			strcat(free_energy_out, ".txt\0");
			/*ofstream out12(free_energy_out, ios::app);
			if(!out12)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out12 << time*tau*pow(10,9) << "  " << endl;
      		
	  		for(int i = 0; i < Ns+1; i++) 
			{
				for (int j = 0; j < Ny+1 ; j++)
				{
				out12 << free_energy[i][j] << '\t';
           			if(j != Ny)
             			out12 << "  ";
           			else 
            			out12 << endl;
				}
			}
			out12 << endl << endl;
			out12.close();*/
			delete []free_energy_out;
//-----------------13---------------------------------------------------    
       		strcpy(div_current, "div_current_\0");
      		strcat(div_current, tt);
      		strcat(div_current, ".txt\0"); 
	  		/*ofstream out13(div_current,ios::app);  
	  		if(!out13)
			cerr<<"Error at the opening!!!.\n"<<endl;
			
			out13 << time*tau*pow(10,9) << "  " << endl; 
			
	  		for (int i = 0; i < Ns + 1; i++)
			{
				for (int j = 0; j < Ny + 1; j++)
				{
				out13 <<  div_j[i][j] << '\t';
           			if(j != Ny)
             			out13 << "  ";
           			else 
            			out13 << endl; //data1 << i*hs*lambda + delta / 2. << " " << j*hy*lambda << " " << mod_PSI(psi1[i][j], psi2[i][j]) << endl;
				}
			}
			
			out13 << endl << endl;
			out13.close();*/
			delete []div_current;
//---------------------------------------------------------------------------		
		}

		if (time_count % (Nt / 10) == 0)
		{
		char *checking;
		checking = new char[50];

		strcpy(checking, "Check_Me\0");
     	strcat(checking, Magnetic);
        strcat(checking, ".txt\0"); 
	    ofstream out(checking,ios::app);  
	  	if(!out)
      	cerr<<"Error at the opening!!!.\n"<<endl;
			out << "Loading... " << time_count * 100 / Nt << "% calculated" << endl;
		out.close();	
		}

	} while (time_count < Nt);

	//memory cleaning
	for (int i = 0; i < Ns + 1; i++)
		delete[] psi_start1[i];
	delete[] psi_start1;

	for (int i = 0; i < Ns + 1; i++)
		delete[] psi_start2[i];
	delete[] psi_start2;

	for (int i = 0; i < Ns + 1; i++)
		delete[] psi1[i];
	delete[] psi1;

	for (int i = 0; i < Ns + 1; i++)
		delete[] psi2[i];
	delete[] psi2;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] kappa_rand[i];
	delete[] kappa_rand;

	for (int i = 0; i < Ns + 1; i++)
		delete[] As[i];
	delete[] As;

	for (int i = 0; i < Ns + 1; i++)
		delete[] Ay[i];
	delete[] Ay;

	for (int i = 0; i < Ns + 1; i++)
		delete[] ReUs[i];
	delete[] ReUs;

	for (int i = 0; i < Ns + 1; i++)
		delete[] ReUy[i];
	delete[] ReUy;

	for (int i = 0; i < Ns + 1; i++)
		delete[] ImUs[i];
	delete[] ImUs;

	for (int i = 0; i < Ns + 1; i++)
		delete[] ImUy[i];
	delete[] ImUy;

	for (int i = 0; i < Ns + 1; i++)
		delete[] fi[i];
	delete[] fi;

	for (int i = 0; i < Ns + 1; i++)
		delete[] fi_half[i];
	delete[] fi_half;

	for (int i = 0; i < Ns + 1; i++)
		delete[] fi_ev[i];
	delete[] fi_ev;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] F_fi[i];
	delete[] F_fi;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] fi_old[i];
	delete[] fi_old;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] F_y[i];
	delete[] F_y;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] j_sc_y[i];
	delete[] j_sc_y;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] j_sc_s[i];
	delete[] j_sc_s;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] j_norm_s[i];
	delete[] j_norm_s;

	for (int i = 0; i < Ns + 1; i++)
		delete[] j_norm_y[i];
	delete[] j_norm_y;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] j_tot_s[i];
	delete[] j_tot_s;

	for (int i = 0; i < Ns + 1; i++)
		delete[] j_tot_y[i];
	delete[] j_tot_y;	
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] free_energy[i];
	delete[] free_energy;
	
	for (int i = 0; i < Ns + 1; i++)
		delete[] B_n[i];
	delete[] B_n;
	
	delete[] s;
	delete[] y;

	delete[] s_dim;
	delete[] y_dim;

	//system("pause");
// 	MPI_Finalize();
	return 0;
}

double ReDss(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUs_mid, double ImUs_mid, double ReUs_prev, double ImUs_prev, double hs)
{
	return ((psiRE_nest*ReUs_mid + psiIM_nest*ImUs_mid - 2.0*psi_mid + psiRE_prev*ReUs_prev - psiIM_prev*ImUs_prev) / (hs*hs));
}

double ImDss(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUs_mid, double ImUs_mid, double ReUs_prev, double ImUs_prev, double hs)
{
	return (((-1.0)*psiRE_nest*ImUs_mid + psiIM_nest*ReUs_mid - 2.0*psi_mid + psiIM_prev*ReUs_prev + psiRE_prev*ImUs_prev) / (hs*hs));
}

double ReDyy(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUy_mid, double ImUy_mid, double ReUy_prev, double ImUy_prev, double hy)
{
	return ((psiRE_nest*ReUy_mid + psiIM_nest*ImUy_mid - 2.0*psi_mid + psiRE_prev*ReUy_prev - psiIM_prev*ImUy_prev) / (hy*hy));
}

double ImDyy(double psiRE_nest, double psiIM_nest, double psi_mid, double psiRE_prev, double psiIM_prev, double ReUy_mid, double ImUy_mid, double ReUy_prev, double ImUy_prev, double hy)
{
	return (((-1.0)*psiRE_nest*ImUy_mid + psiIM_nest*ReUy_mid - 2.0*psi_mid + psiIM_prev*ReUy_prev + psiRE_prev*ImUy_prev) / (hy*hy));
}

double As_value(double s, double y, double B_ind)
{
	return (0.0);
}

double Ay_value(double s, double y, double B_ind, double a)
{
	double R = a / (2. * PI); //**** sleduet li zdes' zamenit' a na a+delta v sootvetstvii s formuloi na strochke 59
	return ((-1.0)*B_ind*R*cos(s / R));
}

double mod_PSI(double psi_RE, double psi_IM)
{
	double res;
	res = pow(psi_RE, 2) + pow(psi_IM, 2);

	if (res > 1.1)
	{
		cout << "Attention!" << endl;
	}

	return res;
}

double max_mass(double **mas, int count_s, int count_y)
{
	double max = mas[0][0];

	for (int i = 0; i < count_s; i++)
	{
		for (int j = 0; j < count_y; j++)
		{
			if (mas[i][j] > max)
			{
				max = mas[i][j];
			}
		}
	}
	return max;
}

void vortex_catch(double time_count, double **psi1, double  **psi2, double *y){
     
	char *vortex_catch;
	vortex_catch = new char[50];

	strcpy(vortex_catch, "current \0");
    strcat(vortex_catch, "j_tr_dim");
    strcat(vortex_catch, ".txt\0 "); 
  
    double w[Ns][Ny], u_temp[Ns][Ny], out_y[Ny]; 

//determining of absolute values of order distribution
     for(int i = 0; i < int((Ns-1)/4); i++)
		for(int j = 0; j < Ny-1; j++) 
            u_temp[i][j] = pow(psi1[i][j], 2) + pow(psi2[i][j], 2);

//auxiliary array initialization     
     for(int i = 0; i < int((Ns-1)/4); i++)
     {
     	for(int j = 0; j < Ny-1; j++)
		 {
         if(u_temp[i][j] < 0.1)
            w[i][j] = y[j];
         else
            w[i][j] = -10;                
        }
     }
	 
//initialization of ourput array
     for(int j=0; j < Ny-1; j++)
         out_y[j] = -10.0-1.0;
		 
//allocation of output array
     for(int i = 0; i < int((Ns-1)/4); i++)
		for(int j = 0; j < Ny-1; j++) 
           {                  
              if(w[i][j]>= -1.0)
                   out_y[j] = w[i][j];
           }

//file openning            
	ofstream out(vortex_catch,ios::app);  
	if(!out)  
    	cerr<<"Error at the opening!!!.\n"<<endl;
    if(time_count > 0)
        out << endl;
                       
    out << time_count*ht*tau*pow(10,9) << '\t';
    
//print in file information about the first part of the cylinder
    for(int i=0; i < Ny-1; i++)
    {
        out << out_y[i] << '\t';
	}        
}


