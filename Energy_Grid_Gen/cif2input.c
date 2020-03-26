#include <stdio.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[])
{
	//This script needs three input string as following: cif file, force field info file and output file name
	//***********
	//This part defines the input parameter
	int FH_SIGNAL = 1;
	double TEMPERATURE = 77;
	int FH_signal = FH_SIGNAL;
	double Temperature = TEMPERATURE;
	int Nmaxa, Nmaxb, Nmaxc;
	// 31 for 32*32*32 and 63 for 61*61*61 grid
	Nmaxa = 31;
	Nmaxb = 31;
	Nmaxc = 31;
	double dL = 0.1;
	double epsilon = 34.200000;
	double sigma = 2.960000;
	double mass_molecule = 2.012000;
	double cutoff = 12.9;
	//This part defines other parameter needed to write the input file
	
	//*************

	//define file varaiable
	FILE *fp1, *fp2;
	int buffersize = 256;
	char str[buffersize];
	//define read-in parameters
	double alpha = 0, beta = 0, gamma = 0;
	double a, b, c;
	int N;
	int i, ii, j;
	double occupancy;
	double cartesian_coordinate[3];
	char atom[2];
	double mass;

	char str1[] = "_cell_length_a";
	char str2[] = "_cell_length_b";
	char str3[] = "_cell_length_c";
	char str4[] = "_cell_angle_alpha";
	char str5[] = "_cell_angle_beta";
	char str6[] = "_cell_angle_gamma";
	char str7[] = "_atom_site_type_symbol";
	char str8[] = "_atom_site_occupancy";
	char empty_line[] = "\n";
	char space[] = " ";
	double x, y, z;
	//done!!!!!

	//parameter read
	//e.g. lattice size and angle
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		fscanf(fp1, "%s", str);
		if (strcmp(str, str1) == 0)
		{
			fscanf(fp1, "%lf", &a);
		}
		else if (strcmp(str, str2) == 0)
		{
			fscanf(fp1, "%lf", &b);
		}
		else if (strcmp(str, str3) == 0)
		{
			fscanf(fp1, "%lf", &c);
		}
		else if (strcmp(str, str4) == 0)
		{
			fscanf(fp1, "%lf", &alpha);
		}
		else if (strcmp(str, str5) == 0)
		{
			fscanf(fp1, "%lf", &beta);
		}
		else if (strcmp(str, str6) == 0)
		{
			fscanf(fp1, "%lf", &gamma);
		}
		else if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
	}
	fclose(fp1);
	//done!!!!!



	// figure out how many line to skip before atomistic information
	int sss1=0, sss2=0;
	int good_signal1 = 0, good_signal2 = 0;
	char extract[buffersize];
	int continue_space;
	int det1;
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
		else
		{
			sss1++;
			if (str[0]==empty_line[0])
			{
				// blank line
			}
			else
			{
				// extract the first word
				i = 0;
				ii = 0;
				while (1)
				{
					if (str[i]==space[0])
					{
						// space
						if (i==0)
						{
							continue_space = 1;
							i++;
						}
						else if (continue_space == 1)
						{
							i++;
						}
						else
						{
							extract[ii] = '\0';
							break;
						}

					}
					else if (str[i]=='\n')
					{
						extract[ii] = '\0';
						break;
					}
					else
					{
						extract[ii] = str[i];
						i++;
						ii++;
						continue_space = 0;
					}
				}
				// done!!!!!
				det1 = 1;
				for (ii=0; ii<strlen(str7); ii++)
				{
					if (str7[ii] != extract[ii])
					{
						det1 = 0;
					}
				}
				if (det1 == 1)
				{
					good_signal1 = 1;
					break;
				}

			}

		}
	}
	fclose(fp1);
	fp1 = fopen(argv[1], "r");
	while (1)
	{
		if ( fgets(str, buffersize, fp1) == NULL)
		{
			break;
		}
		else
		{
			sss2++;
			if (str[0]==empty_line[0])
			{
				// blank line
			}
			else
			{
				// extract the first word
				i = 0;
				ii = 0;
				while (1)
				{
					if (str[i]==space[0])
					{
						// space
						if (i==0)
						{
							continue_space = 1;
							i++;
						}
						else if (continue_space == 1)
						{
							i++;
						}
						else
						{
							extract[ii] = '\0';
							break;
						}

					}
					else if (str[i]=='\n')
					{
						extract[ii] = '\0';
						break;
					}
					else
					{
						extract[ii] = str[i];
						i++;
						ii++;
						continue_space = 0;
					}
				}
				// done!!!!!
				det1 = 1;
				for (ii=0; ii<strlen(str8); ii++)
				{
					if (str8[ii] != extract[ii])
					{
						det1 = 0;
					}
				}
				if (det1 == 1)
				{
					good_signal2 = 1;
					break;
				}
			}
		}
	}
	fclose(fp1);
	int skip_line;
	if ((good_signal1 == 1)&&(good_signal2 == 1))
	{
		if (sss2>=sss1)
		{
			skip_line = sss2;
		}
		else
		{
			skip_line = sss1;
		}
	}
	else
	{
		skip_line = sss1*good_signal1 + sss2*good_signal2;
	}
	// done!!!!!



	// count atom number
	fp1 = fopen(argv[1], "r");
	for (i=0; i<skip_line; i++)
	{
		fgets(str, buffersize, fp1);
	}
	N = 0;
	while (1)
	{
		if ( fgets(str, buffersize, fp1) != NULL)
		{
			fscanf(fp1, "%s ", str);
			if ((str[0]=='l')&&(str[1]=='o')&&(str[2]=='o')&&(str[3]=='p'))
			{
				break;
			}
			else
			{
				N++;
			}
		}
		else
		{
			break;
		}
	}
	fclose(fp1);
	// done!!!

	//store forcefield information before write input file
	//count atom number from forcefiled file
	fp1 = fopen(argv[2], "r");
	fgets(str, buffersize, fp1);
	int N_P = 0;
	while (1)
	{
		if ( fgets(str, buffersize, fp1) != NULL)
		{
			N_P++;
		}
		else
		{
			fclose(fp1);
			break;
		}
	}
	//read atomistic signature from forcefiled file
	char atom_list[N_P][3];
	double signature_list[N_P*3];
	fp1 = fopen(argv[2], "r");
	fgets(str, buffersize, fp1);
	for (i=0; i<N_P; i++)
	{
		// read atom name, sigma, epsilon and mass from forcefield file
		fscanf(fp1, "%s %lf %lf %lf\n", atom_list[i], &signature_list[3*i], &signature_list[3*i+1], &signature_list[3*i+2]);
	}
	fclose(fp1);
	//done!!!

	// write input file
	fp1 = fopen(argv[1], "r");
	fp2 = fopen(argv[3], "w+");
	// write title part
	fprintf(fp2, "File Name:\n");
	fprintf(fp2, "%s\n", argv[1]);

	time_t timer;
	char buffer[26];
	struct tm* tm_info;

	timer = time(NULL);
	tm_info = localtime(&timer);
	strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
	fprintf(fp2, "Created at:\n");
	fprintf(fp2, "%s\n", buffer);

	if (argc > 4){
		fprintf(fp2, "Version:\n");
		fprintf(fp2, "%s\n",argv[4]);

	}else{
		fprintf(fp2, "Version:\n");
		fprintf(fp2, "%s\n", "1.1");
	}

	fprintf(fp2,"Nmaxa Nmaxb Nmaxc:\n");
	// Nmaxa = a/dL + 3;
	// Nmaxb = b/dL + 3;
	// Nmaxc = c/dL + 3;
	fprintf(fp2,"%d\t%d\t%d\n", Nmaxa, Nmaxb, Nmaxc);
	fprintf(fp2,"La Lb Lc dL\n");
	fprintf(fp2,"%lf\t%lf\t%lf\t%lf\n", a, b, c, dL);
	fprintf(fp2,"Alpha Beta Gamma\n");
	fprintf(fp2,"%lf\t%lf\t%lf\n", alpha, beta, gamma);
	fprintf(fp2,"Epsilon(K) Sigma(A) cutoff(A) FH_signal mass(g/mol) Tempearture(K)\n");
	fprintf(fp2,"%lf\t%lf\t%lf\t%d\t%lf\t%lf\n", epsilon, sigma, cutoff, FH_signal, mass_molecule, Temperature);
	fprintf(fp2,"Number of atoms\n");
	fprintf(fp2,"%d\n", N);
	fprintf(fp2,"ID diameter(A) Epsilon(K) mass(g/mol) frac_x frac_y frac_z atom_name\n");
	for (i=0; i<skip_line; i++)
	{
		// skip lines
		fgets(str, buffersize, fp1);
		// done!!!
	}
	if (sss1<sss2)
	{
		//_atom_site_type_symbol shows up first
		for (i=0; i<N; i++)
		{
			fscanf(fp1,"%s ", str);
			fscanf(fp1,"%s", atom);
			fscanf(fp1,"%lf %lf %lf", &x, &y, &z);
			fgets(str, buffersize, fp1);
			// done!!!
			if (atom[1]=='\0')
			{
				// element name has only one character
				for (ii=0; ii<N_P; ii++)
				{
					if (atom_list[ii][1]=='\0')
					{
						if (atom_list[ii][0]==atom[0])
						{
							break;
						}
					}
				}
			}
			else
			{
				// element name has two character
				for (ii=0; ii<N_P; ii++)
				{
					if ((atom_list[ii][0]==atom[0])&&(atom_list[ii][1]==atom[1]))
					{
						break;
					}
				}
			}
			fprintf(fp2,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%s\n", i+1, signature_list[3*ii], signature_list[3*ii+1], signature_list[3*ii+2], x, y, z, atom_list[ii]);
		}
	}
	else if (sss1>sss2)
	{
		//_atom_site_type_symbol shows up later
		for (i=0; i<N; i++)
		{
			fscanf(fp1,"%s ", str);
			fscanf(fp1,"%s ", str);
			fscanf(fp1,"%lf %lf %lf", &x, &y, &z);
			fscanf(fp1,"%s ", str);
			fscanf(fp1,"%s ", str);
			fscanf(fp1,"%s", atom);
			fgets(str, buffersize, fp1);
			// done!!!
			if (atom[1]=='\0')
			{
				// element name has only one character
				for (ii=0; ii<N_P; ii++)
				{
					if (atom_list[ii][1]=='\0')
					{
						if (atom_list[ii][0]==atom[0])
						{
							break;
						}
					}
				}
			}
			else
			{
				// element name has two character
				for (ii=0; ii<N_P; ii++)
				{
					if ((atom_list[ii][0]==atom[0])&&(atom_list[ii][1]==atom[1]))
					{
						break;
					}
				}
			}
			fprintf(fp2,"%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%s\n", i+1, signature_list[3*ii], signature_list[3*ii+1], signature_list[3*ii+2], x, y, z, atom_list[ii]);
		}
	}
	else
	{
		printf("wrong!!!!!!!!!!\t%d\t%d\n", sss1, sss2);
	}

	fclose(fp1);
	fclose(fp2);
}
