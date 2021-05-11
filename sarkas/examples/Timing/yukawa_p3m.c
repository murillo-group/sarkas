#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>


void calculateSTD(double data[], double *mean, double *std) {
    double sum = 0.0, neam, SD = 0.0;
    int i;
    for (i = 0; i < 10; ++i) {
        sum += data[i];
    }
    *mean = sum/10;
    neam = sum / 10;
    for (i = 0; i < 10; ++i) {
        SD += pow(data[i] - neam, 2);
    }
    *std = sqrt(SD / 10);
}


double drand ( double low, double high )
{
    return ( (double)rand() * ( high - low ) ) / (double)RAND_MAX + low;
}
//particle_mesh acceleration update
void particle_mesh(double** pos, double** a_pm, double* Z, int N, double L, double kappa, double G, double hx, double hy, double hz, int Mx, int My, int Mz, int p)
{

  int i, j, k, ipart;
  int ix, iy, iz;
  int ixn, iyn, izn;
  int r_i, r_j, r_k;
  double x, x_2, x_3, x_4, x_5, y, y_2, y_3, y_4, y_5, z, z_2, z_3, z_4, z_5;

  double *wx, *wy, *wz;

  wx = malloc(p*sizeof(double));
  wy = malloc(p*sizeof(double));
  wz = malloc(p*sizeof(double));

  double*** rho_r = malloc(Mz*sizeof(double**));
  for(i = 0; i < Mz; i++) 
  {
     rho_r[i] = malloc(My*sizeof(double*));

     for(j = 0; j < My; j++)
     {
        rho_r[i][j] = malloc(Mx*sizeof(double));
     }
   }

   for(k = 0; k < Mz; k++)
   {
      for(i = 0; i < My; i++)
      {
         for(j = 0; j < Mx; j++)
         {
            rho_r[k][i][j] = 0.0;

         }
      }
   } 

   for(ipart = 0; ipart < N; ipart++)
   {
      
      ix = (int) floor(pos[ipart][0]/hx);
      x = (pos[ipart][0])/hx - (((double) ix) + 0.5);
      x_2 = x*x;
      x_3 = x*x_2;
      x_4 = x*x_3;
      x_5 = x*x_4;

      iy = (int) floor(pos[ipart][1]/hy);
      y = (pos[ipart][1])/hy - (((double) iy) + 0.5);
      y_2 = y*y;
      y_3 = y*y_2;
      y_4 = y*y_3;
      y_5 = y*y_4;
      
      iz = (int) floor(pos[ipart][2]/hz);
      z = (pos[ipart][2])/hz - (((double) iz) + 0.5);
      z_2 = z*z;
      z_3 = z*z_2;
      z_4 = z*z_3;
      z_5 = z*z_4;

      wx[0] = (1.0 - 10.0*x + 40.0*x_2 - 80.0*x_3 + 80.0*x_4 - 32.0*x_5)/3840.0;
      wx[1] = (237.0 - 750.0*x + 840.0*x_2 - 240.0*x_3 - 240.0*x_4 + 160.0*x_5)/3840.0;
      wx[2] = (841.0 - 770.0*x - 440.0*x_2 + 560.0*x_3 + 80.0*x_4 - 160.0*x_5)/1920.0;
      wx[3] = (841.0 + 770.0*x - 440.0*x_2 - 560.0*x_3 + 80.0*x_4 + 160.0*x_5)/1920.0;
      wx[4] = (237.0 + 750.0*x + 840.0*x_2 + 240.0*x_3 - 240.0*x_4 - 160.0*x_5)/3840.0;
      wx[5] = (1.0 + 10.0*x + 40.0*x_2 + 80.0*x_3 + 80.0*x_4 + 32.0*x_5)/3840.0;
      

      wy[0] = (1.0 - 10.0*y + 40.0*y_2 - 80.0*y_3 + 80.0*y_4 - 32.0*y_5)/3840.0;
      wy[1] = (237.0 - 750.0*y + 840.0*y_2 - 240.0*y_3 - 240.0*y_4 + 160.0*y_5)/3840.0;
      wy[2] = (841.0 - 770.0*y - 440.0*y_2 + 560.0*y_3 + 80.0*y_4 - 160.0*y_5)/1920.0;
      wy[3] = (841.0 + 770.0*y - 440.0*y_2 - 560.0*y_3 + 80.0*y_4 + 160.0*y_5)/1920.0;
      wy[4] = (237.0 + 750.0*y + 840.0*y_2 + 240.0*y_3 - 240.0*y_4 - 160.0*y_5)/3840.0;
      wy[5] = (1.0 + 10.0*y + 40.0*y_2 + 80.0*y_3 + 80.0*y_4 + 32.0*y_5)/3840.0;


      wz[0] = (1.0 - 10.0*z + 40.0*z_2 - 80.0*z_3 + 80.0*z_4 - 32*z_5)/3840.0;
      wz[1] = (237.0 - 750.0*z + 840.0*z_2 - 240.0*z_3 - 240.0*z_4 + 160.0*z_5)/3840.0;
      wz[2] = (841.0 - 770.0*z - 440.0*z_2 + 560.0*z_3 + 80.0*z_4 - 160.0*z_5)/1920.0;
      wz[3] = (841.0 + 770.0*z - 440.0*z_2 - 560.0*z_3 + 80.0*z_4 + 160.0*z_5)/1920.0;
      wz[4] = (237.0 + 750.0*z + 840.0*z_2 + 240.0*z_3 - 240.0*z_4 - 160.0*z_5)/3840.0;
      wz[5] = (1.0 + 10.0*z + 40.0*z_2 + 80.0*z_3 + 80.0*z_4 + 32.0*z_5)/3840.0;
      
      izn = iz;
      for(k = 0; k < p; k++)
      {
         if(izn < 0) {
           r_k = izn + Mz;
         }
         else if(izn > (Mz - 1)) {
           r_k = izn - Mz;
         }
         else {
           r_k = izn;
         }  
   
         iyn = iy;
         for(i = 0; i < p; i++)
         {            
            if(iyn < 0) {
              r_i = iyn + My;
            }
            else if(iyn > (My - 1)) {
              r_i = iyn - My;
            }
            else {
              r_i = iyn;
            }  

            ixn = ix;
            for(j = 0; j < p; j++) 
            {
               if(ixn < 0) {
                 r_j = ixn + Mx;
               }
               else if(ixn > (Mx - 1)) {
                 r_j = ixn - Mx;
               }
               else {
                 r_j = ixn;
               }

               rho_r[r_k][r_i][r_j] += Z[ipart] * wz[k] * wy[i] * wx[j];
               ixn += 1;
             } 

            iyn += 1;  
         }
       
          izn += 1;
       }

  
   }

   // for(k = 0; k < 3; k++)
   // {
   //    for(i = 0; i < 3; i++)
   //    {
   //       for(j = 0; j < 3; j++)
   //       {
   //          printf("%lf\t", rho_r[k][i][j]);

   //       }
   //       printf("\n");
   //    }
   //    printf("\n");
   // } 

   // printf("rho(5,5,5) = %f\n",rho_r[5][5][5]);

   double sum_charge = 0.0;
   for(k = 0; k < Mz; k++)
   {
      for(i = 0; i < My; i++)
      {
         for(j = 0; j < Mx; j++)
         {
            sum_charge += rho_r[k][i][j];
         }
      }
    }

   // printf("sum(Z) = %lf\n", sum_charge);

   free(rho_r); free(wx); free(wy); free(wz);

}

// acceleration update
void particle_particle(double** pos, double** a, double* Z, int n, double L, double kappa, double G, double rc)
{

    double Lx, Ly, Lz;
    double rc_x, rc_y, rc_z;
    int Lxd, Lyd, Lzd;
    int Ncell;
    int c, cx, cy, cz;
    int c_N, cx_N, cy_N, cz_N;
    int cx_shift, cy_shift, cz_shift;
    double rshift[3];
    double dx, dy, dz, r;
    double f1, f2, f3, fr;
    double Z_ij;
    int i, j;
    int counter;

    int *head;
    int *ls;
    int size;

    int empty = -50;
    int d = 3;
    //printf("empty cell = %d", empty);
    
    double pi = 3.141592653589793;

    double U_s_r = 0;
    //printf("U = %lf\n", U_s_r);

    Lx = L;
    Ly = L;
    Lz = L;

    Lxd = (int) floor(Lx/rc);
    Lyd = (int) floor(Ly/rc);
    Lzd = (int) floor(Lz/rc);

    rc_x = Lx/Lxd;
    rc_y = Ly/Lyd;
    rc_z = Lz/Lzd;

    Ncell = Lxd * Lyd * Lzd;

    //printf("Ncell = %d\n", Ncell);
    //printf("rcx, rcy, rcz = %.11lf, %.11lf, %.11lf\n", rc_x, rc_y, rc_z);
    //printf("Lxd, Lyd, Lzd = %d, %d, %d\n", Lxd, Lyd, Lzd);
    //printf("Lx, Ly, Lz = %lf, %lf, %lf\n", Lx, Ly, Lz); 
    //printf("kappa, Gew, rc = %lf, %lf, %lf\n",kappa, G, rc);
    
    size = sizeof(int);
    head = (int *)malloc(Ncell*size);
    ls = (int *)malloc(n*size);

    for(i = 0; i < Ncell; i++)
    {
       head[i] = empty;
       //printf("%d %d\n", head[i], i+1);
    }

    for(i = 0; i < n; i++)
    {
       ls[i] = i;
       //printf("%d %d\n", head[i], i+1);
    }
    
    for(i = 0; i < 3; i++)
    {
       rshift[i] = 0;
       //printf("%lf ", rshift[i]);
    }
    
    for(i = 0; i < n; i++)
    {
       for(j = 0; j < 3; j++)
       {
          a[i][j] = 0;

          //printf("%.8lf ",a[i][j]); //Use lf format specifier, \n is for new line
          //if (j==2) printf("\n");
       }
    }

    for(i = 0; i < n; i++)
    {
       cx = (int) floor(pos[i][0]/rc_x);
       cy = (int) floor(pos[i][1]/rc_y);
       cz = (int) floor(pos[i][2]/rc_z);
       //if(i == 0) printf("cx, cy, cz = %d, %d, %d\n", cx, cy, cz);
       c = cx + cy*Lxd + cz*Lxd*Lyd;
       //if(i > 990) 
       //{
         //printf("c = %d\n",c);
         //printf("i, head[c], ls[i] = %d, %d, %d\n",i, head[c], ls[i]);
       //}

       //if(c == 125){ printf("c = %d, ipart = %d, x = %f, y = %f, z = %f\n",c, i, pos[i][0], pos[i][1], pos[i][2]);} 
 //       if(c == 120){ printf("c = %d, ipart = %d, x = %f, y = %f, z = %f\n",c, i, x[i], y[i], z[i]);}
 //              //if(c == 89){ printf("c = %d, ipart = %d, x = %f, y = %f, z = %f\n",c, i, x[i], y[i], z[i]);} 
 //                     //if(c == 120){ printf("c = %d, ipart = %d, x = %f, y = %f, z = %f\n",c, i, x[i], y[i], z[i]);} 

       ls[i] = head[c];
       head[c] = i; 
       
       //if(i > 990) 
       //{
         //printf("c = %d\n",c);
         //printf("i, head[c], ls[i] = %d, %d, %d\n",i, head[c], ls[i]);
       //}
    }

    counter = 0;
    for(cx = 0; cx < Lxd; cx++)
    {
       for(cy = 0; cy < Lyd; cy++)
       {
    	  for(cz = 0; cz < Lzd; cz++)
          {

             c = cx + cy*Lxd + cz*Lxd*Lyd;
             counter += 1;
             //printf("c = %d\n", c);
             
             for(cz_N = cz-1; cz_N < cz+2; cz_N++)
             {
                // if(cz_N < 0) {
                //     cz_shift = Lzd;
                //     rshift[2] = -Lz;
                //   }      
                //   else if(cz_N >= Lzd) {
                //     cz_shift = -Lzd;
                //     rshift[2] = Lz;
                //   }
                //   else {
                //     cz_shift = 0;
                //     rshift[2] = 0;
                //   }

                  cz_shift = 0 + Lzd * (cz_N < 0) - Lzd * (cz_N >= Lzd);
                  rshift[2] = 0.0 - Lz * (cz_N < 0) + Lz*(cz_N >= Lzd);

                for(cy_N = cy-1; cy_N < cy+2; cy_N++)
                {        

                      // if(cy_N < 0) {
                      //   cy_shift = Lyd;
                      //   rshift[1] = -Ly;
                      // }      
                      // else if(cy_N >= Lyd) {
                      //   cy_shift = -Lyd;
                      //   rshift[1] = Ly;
                      // }
                      // else {
                      //   cy_shift = 0;
                      //   rshift[1] = 0;
                      // }
                      
                      cy_shift = 0 + Lyd * (cy_N < 0) - Lyd * (cy_N >= Lyd);
                      rshift[1] = 0.0 - Ly * (cy_N < 0) + Ly * (cy_N >= Lyd);


                   for(cx_N = cx-1; cx_N < cx+2; cx_N++)
                   {

                   
                      // if(cx_N < 0) {
                      //   cx_shift = Lxd;
                      //   rshift[0] = -Lx;
                      // }      
                      // else if(cx_N >= Lxd) {
                      //   cx_shift = -Lxd;
                      //   rshift[0] = Lx;
                      // }
                      // else {
                      //   cx_shift = 0;
                      //   rshift[0] = 0;
                      // }

                      cx_shift = 0 + Lxd * (cx_N < 0) - Lxd * (cx_N >= Lxd);
                      rshift[0] = 0.0 - Lx * (cx_N < 0) + Lx * (cx_N >= Lx);



                      c_N = (cx_N + cx_shift) + (cy_N + cy_shift)*Lxd + (cz_N + cz_shift)*Lxd*Lyd;
                      //if(counter < 5) printf("c, c_N = %d, %d\n", c, c_N);
                      //if(c == 124) {printf("c = %d, c_N = %d, xsh = %lf, ysh = %lf, zsh = %lf\n", c, c_N, rshift[0], rshift[1], rshift[2]);}
                      
                      i = head[c];

                      while(i != empty) {

                           j = head[c_N];

                           while(j != empty) {

                                if(i < j) {

                                  //if(c == 0) {printf("c = %d, c_i = %d, c_j = %d\n", c, i, j);}
                                  dx = pos[i][0] - (pos[j][0] + rshift[0]);
                                  dy = pos[i][1] - (pos[j][1] + rshift[1]);
                                  dz = pos[i][2] - (pos[j][2] + rshift[2]);
                                  r = sqrt(pow(dx,2) + pow(dy,2) + pow(dz,2));

                                  if(r < rc) {

                                   Z_ij = Z[i] * Z[j];

                                   U_s_r = U_s_r + Z_ij * (0.5/r) * (exp(kappa*r) * erfc(G*r + 0.5*kappa/G) + exp(-kappa*r) * erfc(G*r - 0.5*kappa/G));

                                   f1 = (0.5/pow(r,2)) * exp(kappa*r) * erfc(G*r + 0.5*kappa/G) * (1.0 - kappa*r);                                    
                                   f2 = (0.5/pow(r,2)) * exp(-kappa*r) * erfc(G*r - 0.5*kappa/G) * (1.0 + kappa*r);
                                   f3 = (G/(sqrt(pi)*r)) * (exp(-pow((G*r + 0.5*kappa/G),2)) * exp(kappa*r) + exp(-pow((G*r - 0.5*kappa/G),2)) * exp(-kappa*r) );
                                   fr = Z_ij*(f1 + f2 + f3);                                   
                                  
                                   a[i][0] += fr*dx/r;
                                   a[i][1] += fr*dy/r;
                                   a[i][2] += fr*dz/r;
                                   
                                   a[j][0] -= fr*dx/r;
                                   a[j][1] -= fr*dy/r;
                                   a[j][2] -= fr*dz/r;
                                   
                                  }

                                }
                                
                                j = ls[j];
                           }

                           i = ls[i];

                      }

                   }
                 }
              }


          }
       }
     }

    // printf("U = %lf\n", U_s_r);

    free(head); free(ls);
}


int main(int argc, char *argv[])
{

  
  const double PI = 3.14159265358979323846;
  int i;
  int j;

  double diff[10];
  double mean, std;

  struct timespec start, end;
  double billion = 1000000000; 
  FILE *file;

  
  // file = fopen(argv[2], "r");
  int N = atoi(argv[1]);
  // double Gew = atof(argv[3]);
  // double rc = atof(argv[4]);

  //file = fopen("pos_1e6.txt", "r");
  // int N = 10000;
  double Gew = 0.56;
  double rc = 6.2466;

  double kappa = 0.0;
  double L = cbrt(4.0 * PI/3.0)*cbrt((double) N);
  printf("L = %3.12lf\n", L);
  printf("Gew = %.4lf\n", Gew);
  printf("rc = %1.4lf\n", rc);

  // particle-mesh part
  int Mx = 32;
  int My = 32;
  int Mz = 32;

  int p = 6; // order of B-splines

  double hx = L/Mx;
  double hy = L/My;
  double hz = L/Mz;

  printf("hx = %lf, hy = %lf, hz = %lf\n", hx, hy, hz);

  double** pos = malloc(N*sizeof(double*)); 
  for(i=0;i<N;++i)
     pos[i] = malloc(3*sizeof(double));


  //FILE *file;
  //file = fopen("pos_1e6.txt", "r");

 for(i = 0; i < N; i++)
  {
      for(j = 0; j < 3; j++) 
      {
        // if (!fscanf(file, "%lf", &pos[i][j])) 
        //   break;
        pos[i][j] = drand(0.0, L);
       }

   }
  // fclose(file);

  // allocating particle_particle component of accelerations
  double** acc_pp = malloc(N*sizeof(double*));
  for(i=0;i<N;++i)
  acc_pp[i] = malloc(3*sizeof(double));

  for(i = 0; i < N; i++)
  {
     for(j = 0; j < 3; j++)
     {
        acc_pp[i][j] = 0;
     }
   }

  // allocating particle_particle component of accelerations
  double** acc_pm = malloc(N*sizeof(double*));
  for(i=0;i<N;++i)
  acc_pm[i] = malloc(3*sizeof(double));

  for(i = 0; i < N; i++)
  {
     for(j = 0; j < 3; j++)
     {
        acc_pm[i][j] = 0;
     }
   }

  double* Z = malloc(N*sizeof(double));
      for(i = 0; i < N; i++)
      {
         Z[i] = 1;
      }

  for (i=0; i<10; i++){ diff[i] = 0;
  }
  for (i = 0; i < 10; i++){
    clock_gettime(CLOCK_MONOTONIC, &start);	/* mark start time */
	
    particle_particle(pos, acc_pp, Z, N, L, kappa, Gew, rc);

    clock_gettime(CLOCK_MONOTONIC, &end);	/* mark the end time */

    diff[i] = (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_nsec - start.tv_nsec)/1000.0;

}
  calculateSTD(diff, &mean, &std);
  printf("PP elapsed time = %lf, %lf micro-seconds\n", mean*1000.0, std*1000.0 );

  //

  // clock_gettime(CLOCK_MONOTONIC, &start);       /* mark start time */

  // particle_mesh(pos, acc_pm, Z, N, L, kappa, Gew, hx, hy, hz, Mx, My, Mz, p);

  // clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */

  // diff[0] = (end.tv_sec - start.tv_sec)*1000000.0 + (end.tv_nsec - start.tv_nsec)/1000.0;
  // printf("PM elapsed time = %lf micro-seconds\n", diff);

  // for(i = 0; i < 10; i++)
  // {
  //    //for(j = 0; j < 3; j++)
  //    //{
  //      // printf("%.8lf ",acc_pp[i][j]);
  //       //if (j==2) printf("\n");
  //    //}
  //    printf("%d, %.8lf, %.8lf, %.8lf\n",i, acc_pp[i][0], acc_pp[i][1], acc_pp[i][2]);
  // }

  printf("----------------\n");
  
  int chk = 467;
  // printf("%d, %.8lf, %.8lf, %.8lf\n",chk, acc_pp[chk][0], acc_pp[chk][1], acc_pp[chk][2]);
  
  printf("----------------\n");

  // for(i = (N-10); i < N; i++)
  // {
  //    for(j = 0; j < 3; j++)
  //    {
  //       printf("%.8lf ",acc_pp[i][j]);
  //       if (j==2) printf("\n");
  //    }
  // }

  free(Z); free(pos); free(acc_pp);

  return 0;

 }
