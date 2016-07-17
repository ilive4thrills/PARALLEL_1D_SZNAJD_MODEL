#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
 
/* Here is what the command line implementation should look like: */
/* mpiexec -n 2 ./mpi2 TOTALSIZE P_VALUE NUM_OF_RUNS */
/*                  [0]    [1]      [2]      [3]         */
 
#define XSIZE          100
#define UP             1
#define DOWN           0

/* lattice initialized to have only down sites */
void init_array(int sublattice[],int sublatticesize);

/* lattice’s up sites are randomly distributed */
void randomization_init(int sublattice[], int * c, int * number_up, 
int * number_down, int sublatticesize);

/* pick a random site within the sub-lattice */
int random_num_interval (int sublatticesize);  

/* pick pair, set up updating conditions (for interior or boundary locations), and update sites */
void convert_to_global_indices_and_update(int sublattice[], int array[], MPI_Win 
sublattice_win, MPI_Win sublattice_btm, MPI_Win sublattice_top, int * localstep);

/* function called when sites are not near the sub-lattice boundaries */
void update_near_neighbor(int lattice[], int array[]);

/* determine process topology */
void neighbor_determination(int taskid, int * low_neighbor, int * high_neighbor, int numtasks);

/* function called when sites to be updated are near the sub-lattice boundaries */
void update_near_neighbor_BCs(int sublattice[], int array[], MPI_Win sublattice_win, MPI_Win sublattice_btm, MPI_Win sublattice_top, int * localstep);

/* random number generator, used to pick random sites in lattices */
long random_at_most(long max);

/* count the number of up and down sites in a sub-lattice */
void updown_ct(int * up, int * down, int state, int sublatticesize);

/* establish the indices of the site pair and the immediate neighbors */
int global_assignments(int global_ss, int * global_ss2, int * global_ln, int * global_rn, int latticesize);

/* create the files for data writing, give them the proper file names */
void create_and_open_files(FILE * output, FILE * output_MCS, FILE * output_m, char * latticesize, char * p, char * N);

/* helper function used to establish files names */
void concatenate_strings(char * string, char * message, char * latticesize, char * p, char * N);

/* used to randomly select a processor when leftover down sites need to be distributed */
int return_rand_proc(int numtasks);
 
void main (int argc, char *argv[])
{
if (argc != 4 ) {  /* check that program is being properly used */
        printf("You forgot to include one or more arguments.\n"); 
}

struct rlimit rlim = { RLIM_INFINITY, RLIM_INFINITY };
if ( setrlimit(RLIMIT_STACK, &rlim) == -1 ) {                       /* increase stack size */
        perror("setrlimit error");
        exit(1);
}
 
int random_time = time(NULL);    / *seed random number generator with Epoch time */
srand(random_time);             /* continuation of seeding the pseudorandom number generator */

/* NOTE: unmarked variables serve the same purpose as their serial program counterparts */
int     numtasks;        /* number of processors working on the simulation */
int     sublatticesize;  /* number of sites in the sub-lattice */
int     latticesize;        /* the snumber of sites in the original lattice under study */
double p;
int     N;
int *   sublattice;
int *   local_c;  /* number of down sites in a particular sub-lattice */
int     taskid;      /* identifier unique to each process (just an integer) */
int     i;
int     j;
int     x;
int     c;                      /* several for-loop tracking variables */
int     t;
int     k;
int     one = 1;
int     proc;
int     number_up;
int     number_down;
int     low_neighbor;             /* low neighbor is the “left” process in the process topology */
int     high_neighbor;         /* high neighbor is the “right” process in the process topology */
int     number_up_sum;
int     localstep;   /* a single 1/L MCS, done by one processor */
int     colltd_numup;    /* number of up sites (combined from all processors) */
int     colltd_numdn;  /* number of down sites (combined from all processors) */
int     MCS_count;
int     rsn;
int     c1;
int     remainder = 0;
double  x_value;
double  m;
float   sum;
double  globetime;
FILE *  output = 0;               /*output file pointer for all E(x,L) information*/
FILE *  output_MCS = 0;           /*output file pointer for all MCS information*/
FILE *  output_m = 0;             /*output file pointer for all m(t) information */

/* various numbers that each process should know, such as neighbors, numup, numdown, etc. */
int     array[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};   

clock_t start = clock();  /* grab simulation start time (want to include MPI initializations) */
 
/***** MPI Initializations *****/
MPI_Init(&argc, &argv);                /* initialize the MPI context */

/* use MPI_COMM_WORLD to determine number of tasks (processors) */
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);         

/* find out processor identifier */
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

/* incorporate some error handling protocols into the MPI environment */
MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
 
latticesize = atoi(argv[1]);
sublatticesize = atoi(argv[1])/numtasks;
p = atof(argv[2]);
N = atoi(argv[3]);
lattice = (int *)malloc(numtasks*sublatticesize*sizeof(int));
 
MPI_Win sublattice_win;    /* creation of MPI window object variables */
MPI_Win sublattice_btm;
MPI_Win sublattice_top;
MPI_Win local_c_win;

/* allocate memory locations for sub-lattice and local up-amount variables */ 
MPI_Alloc_mem(sizeof(int)*sublatticesize, MPI_INFO_NULL, &sublattice);   
MPI_Alloc_mem(sizeof(int)*1, MPI_INFO_NULL, &local_c);
 
/* creation of the MPI windows overlaid on the allocated memory ranges */
MPI_Win_create(sublattice, sublatticesize*sizeof(int), sizeof(int), MPI_INFO_NULL, 
	MPI_COMM_WORLD, &sublattice_win);
MPI_Win_create(sublattice, 3*sizeof(int), sizeof(int), 
	MPI_INFO_NULL,  MPI_COMM_WORLD, &sublattice_btm);
MPI_Win_create(&(sublattice[sublatticesize - 3]), 3*sizeof(int), sizeof(int), MPI_INFO_NULL, 
	MPI_COMM_WORLD, &sublattice_top);
MPI_Win_create(local_c, 1*sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, 
	&local_c_win);
 
/* create and open the data output files */
if (taskid == 0) {
        create_and_open_files(output, output_MCS, output_m, argv[1], argv[2], argv[3]);
}
 
/* determine the processor’s place within the process topology */
neighbor_determination(taskid, &low_neighbor, &high_neighbor, numtasks);  
 
array[0] = taskid;
array[1] = numtasks;
array[2] = sublatticesize;
array[3] = 0;   /* number_up */
array[4] = 0;    /* number_down */
array[5] = 0;    /* ln (C) */
array[6] = 0;    /* ss/rsn (A) */
array[7] = 0;    /* rn (D)*/
array[8] = low_neighbor;
array[9] = high_neighbor;
array[10] = latticesize;
array[11] = 0;       /* rsn */

MPI_Barrier(MPI_COMM_WORLD);
 
for (x = 25; x < 26; x++) {
        x_value = ((double)x)/((double)XSIZE);
        c = x_value*latticesize;   /* total number of up sites initially in the original lattice */
        if (c % numtasks == 0) {
     	*local_c = c/numtasks;   /* all processes get an equal number of down sites initially */
        } else {
     	  remainder = c % numtasks;
     	  *local_c = (c - remainder)/numtasks; /* randomly assign the leftover sites */
     	  if (taskid == 0) {           /* have processor 0 disperse the random sites */
             	for (t = 0; t < remainder; t++) {
             	  proc = return_rand_proc(numtasks);  /* send an up site to a random process */
/* processor 0 accesses other processors’ local c values, increments them */
             	  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, proc, 0, local_c_win);
             	  MPI_Accumulate(&one, 1, MPI_INT, proc, 0, 1, 
MPI_INT, MPI_SUM, local_c_win);
             	  MPI_Win_unlock(proc, local_c_win);
             	} 
                }  
        }
        number_up_sum = 0;
        for (i = 0; i < N; i++) {
     	       MPI_Barrier(MPI_COMM_WORLD); /* ensure all processors are on same trial */
                c1 = *local_c;     
                number_up = 0, number_down = 0;
                MCS_count = 0, localstep = 0;
                init_array(sublattice, sublatticesize);
                randomization_init(sublattice, &c1, &number_up, &number_down, sublatticesize);
                colltd_numup = 0;
                colltd_numdn = 0;  /* initial number of up/down sites (all processors combined) */
                do {
                       convert_to_global_indices_and_update(sublattice, array, 
sublattice_win, sublattice_btm, sublattice_top, &localstep);
                        number_up = 0, number_down = 0;
                        for (k = 0; k < sublatticesize; k++) { /* find number up and down in sub-lattice */
                                if (sublattice[k] == UP) number_up += 1;
                                if (sublattice[k] == DOWN) number_down += 1;
                        }
                        MPI_Allreduce(&number_up, &colltd_numup, 1, MPI_INT, MPI_SUM, 
			MPI_COMM_WORLD);
                        m = ((double)(colltd_numup - colltd_numdn))/((double)latticesize);
		/* processor 0 records the m(t) data */ 
                        if (tasked == 0) fprintf(output_m, “%3f %d %d\n”, m, i, latticesize); 
                } while ((colltd_numup < latticesize) && (colltd_numup > 0));
                if (colltd_numup == latticesize) {
                        number_up_sum = number_up_sum + 1;
                }
        }
        /* pofx reduction, print results to screen */
        if (taskid == 0) { 
printf("%f %f\n", x_value, (double)number_up_sum/(double)N);
        }
}
 
/*closure stuff here. */
clock_t finish = clock();

/*compute how long the simulation took in hours/minutes/seconds, print that to screen*/
double sim_time = (((double) (finish - start)) / (CLOCKS_PER_SEC)); 

/* write the actual time of simulation to the screen*/
printf("\nTime elapsed was: %3f s = %3f minutes, done by task #%d\n", sim_time, sim_time / (60),taskid);  

free(lattice);
MPI_Free_mem(sublattice);  /* free the memory used for the window objects */
MPI_Free_mem(local_c);
MPI_Win_free(&sublattice_win);
MPI_Win_free(&sublattice_btm);
MPI_Win_free(&sublattice_top);
MPI_Win_free(&local_c_win);
 
if (taskid == 0) {
        close(output);
        close(output_MCS);   /* close the file pointers for the data files */
        close(output_m);
}
MPI_Finalize();
}  
/* end of main */
 
/* perform pair selection, map A,B,C,and D to indices, call applicable site updating function */
void convert_to_global_indices_and_update(int sublattice[], int array[], MPI_Win sublattice_win, MPI_Win sublattice_btm, MPI_Win sublattice_top, int * localstep)
{
        int sublatticesize = array[2]; 
        int ss = random_at_most((sublatticesize - 1)); /* pick random site within lattice */  
        int ss2 = 0, rn = 0, ln = 0;  /* initialize the values of the site pair and neighbor indices */
       
/* find indices of site pair and neighbors */ 
        int flag = global_assignments(ss, &ss2, &ln, &rn, sublatticesize); 
        array[5] = ln;    
        array[6] = ss;    
        array[7] = rn;    
 
        if (flag > 1) { /* updating will be done near sub-lattice boundaries, locking needed */
                update_near_neighbor_BCs(sublattice, array, sublattice_win, 
       sublattice_btm, sublattice_top, localstep);
        }
        else {  /* update a normal interior site */

                if (sublattice[ss] == sublattice[ss2]) {
                        update_near_neighbor(sublattice, array);
                }
        }
}
 
/* set the process topology so that periodic boundary conditions are sustained */
void neighbor_determination(int taskid, int * low_neighbor, int * high_neighbor, int numtasks)
{ 
        if (taskid == 0) {
                *low_neighbor = numtasks - 1;
                *high_neighbor = 1;
        } else if (taskid == (numtasks - 1)) {
                *low_neighbor =  taskid - 1;
                *high_neighbor = 0;
        } else {
                *low_neighbor = taskid - 1;
                *high_neighbor = taskid + 1;
        }
}

/* the function which does the necessary locking and updating of sites near the boundaries */
void update_near_neighbor_BCs(int sublattice[], int array[], MPI_Win sublattice_win, MPI_Win sublattice_btm, MPI_Win sublattice_top, int * localstep)
{      
/* NOTE: ANY VARIABLE ASSOCIATED WITH A WINDOW SHOULD BE LOCKED WHENEVER IT IS PART OF AN UPDATE SELECTION, EVEN IF IT IS NOT BEING UPDATED. WE DON'T WANT OUR ORIGINAL SITE PAIRS TO BE MODIFIED WHILE THEY ARE MODIFYING OTHER SITES. */
        int taskid =  array[0];
        int numtasks =  array[1];
        int sublatticesize = array[2];
        int up =  array[3];
        int down = array[4];           /* diagram of the four-site panel: [ln][rsn/ss][ss2][rn] */
        int ln = array[5];                 /* Alternatively, [C][A][B][D] */
        int ss = array[6];
        int rn = array[7];
        int low_neighbor = array[8];
        int high_neighbor = array[9];
        int latticesize = array[10];
        int rsn = array[11];
        int index_low = array[12];
        int index_high = array[13];
       
        int stateofmaxindexoflowerprocess[1];
        int stateofzeroindexinthisprocess[1];
        int stateofoneindexinthisprocess[1];       /* site-state tracking variables */
        int stateofmaxindexinthisprocess[1];
        int stateofzeroindexofhigherprocess[1];
        int stateofoneindexinhigherprocess[1];
       
	/* note: processor X=>([…][…]) indicates that the sites (square brackets) in the 
parentheses belong to Processor X. The processor which ss/rsn/A belongs to is called 
processor I */
           
       if (ss == 0) {  /* processor S=>([…][C])([ss/A][B][D])<=Processor I */
	    /* this first lock allows for A and B to stay the same */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_btm);
	    /* this second lock allows for C to stay the same before being updated */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, low_neighbor, 0, sublattice_top); 

                if (sublattice[ss] == sublattice[ss+1]) { /* make sure UP1 is satisfied */
                        *localstep += 1;  /* increment the MCS count */
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, low_neighbor, 2, 1, MPI_INT, 
			sublattice_top);  /* the actual remote updating operation */
                        sublattice[rn] = sublattice[ss]; 
                }
                MPI_Win_unlock(taskid, sublattice_btm);
                MPI_Win_unlock(low_neighbor, sublattice_top);
        }
        if (ss == 1) { /* processor J=>([…][…])([C][A][B][D])<=Processor I */
	   /* only one lock is needed (for Processor I) */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_btm);  
                if (sublattice[ss] == sublattice[ss + 1]) {
                        *localstep += 1;
                        sublattice[ln] = sublattice[ss];
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, taskid, 0,1, MPI_INT, sublattice_btm); 
                        sublattice[rn] = sublattice[ss];    
                }
                MPI_Win_unlock(taskid, sublattice_btm);
        }
        if (ss == 2) {  /* processor J=>([…][…])([0][C][A][B][D])<=Processor I */

	
              /* same locking scenario as when ss = 1 */
  	 MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_btm); 
                if (sublattice[ss] == sublattice[ss + 1]) {
                        *localstep += 1;
                        sublattice[ln] = sublattice[ss];    
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, taskid, 1, 1, MPI_INT, sublattice_btm); 
                        sublattice[rn] = sublattice[ss];    
                }
                MPI_Win_unlock(taskid, sublattice_btm);  
        }
        if (ss == (index_high - 2)) { /* processor I=>([C][A][B][D])([…][…])<=Processor J */
	/* all sites of interest are in processor I, so only one window needs to be locked */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_top); 
                if (sublattice[ss] == sublattice[ss + 1]) {
                        *localstep += 1;
                        sublattice[ln] = sublattice[ss];    
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, taskid, 2, 1, MPI_INT, sublattice_top);   
                        sublattice[index_high] = sublattice[ss];
                }
                MPI_Win_unlock(taskid,sublattice_top); 
        }
        if (ss == (index_high - 1)) { /* processor I=>([C][A][B])([D][…])<=Processor J */
	    /* this first lock ensures that A,B, and C remain stable */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_top);  
	   /* this second lock ensures that D remains stable for the possible upcoming update */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, high_neighbor, 0, sublattice_btm);  
         
                if ((sublattice[ss] == sublattice[ss + 1])) {
                        *localstep += 1;
                        sublattice[ln] = sublattice[ss];
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, high_neighbor, 0, 1, MPI_INT, 
			sublattice_btm);
                }
                MPI_Win_unlock(taskid, sublattice_top);
                MPI_Win_unlock(high_neighbor, sublattice_btm);
        }
        if (ss == (index_high)) {  /* processor I=>([C][A])([B][D])<=Processor J */
	    /* the first lock allows for A and C to be kept stable */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, taskid, 0, sublattice_top);   
	    /* the second lock allows B and D to remain stable */
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, high_neighbor, 0, sublattice_btm);   
                /* check to see if site B (in processor J) agrees with A */
    MPI_Get(&(stateofzeroindexofhigherprocess[0]), 1, MPI_INT, high_neighbor, 0, 1, 
		MPI_INT, sublattice_btm);
                if ((stateofzeroindexofhigherprocess[0] == sublattice[ss])) {
                        *localstep += 1;
                        sublattice[ln] = sublattice[ss]; 
                  
            /* set D to the same state as A and B (/
                        MPI_Put(&(sublattice[ss]), 1, MPI_INT, high_neighbor, 1, 1, MPI_INT, 
			sublattice_btm);
                }
                MPI_Win_unlock(taskid, sublattice_top);
                MPI_Win_unlock(high_neighbor, sublattice_btm);
        }
}

/* count the number of up and down sites after each pair selection in a sub-lattice */ 
void updown_ct(int * up, int * down, int state, int sublatticesize)
{ 
        if (state == DOWN) {
                *down += 1;
                *up = sublatticesize - *down;
        }
        if (state == UP) {
                *up += 1;
                *down = (sublatticesize) - *up;
        }
}
 
/* map A,B,C, D to the proper array indices (and processors) so that periodic boundary 
conditions are obeyed */
int global_assignments(int ss, int * ss2, int * ln, int * rn, int sublatticesize)  
{
        if ((ss > 2) && (ss < (sublatticesize - 3))) {   /* an interior site, no locking needed */
                *ss2 = ss + 1;
                *rn = ss + 2;
                *ln = ss - 1;
                return 1;
        }      
        if (ss == 0) {                           
                *ln = sublatticesize - 1;
                *ss2 = 1;                      /* the left neighbor belongs to the “left” processor */
                *rn = 2;
                return 2;
        }
        if (ss == 1) {
                *ss2 = ss + 1;
                *rn = ss + 2;          /* normal neighbor assignment, but locking will be necessary */
                *ln = ss – 1;
                return 3;      
        }
        if (ss == 2) {
                *ss2 = ss + 1;
                *rn = ss + 2;             /* same as above */
                *ln = ss - 1;
                return 4;      
        }

/* one of the site pair, and the right neighbor site belongs to the “right” processor */
        if (ss == (sublatticesize - 1)) {    
                *ln = ss - 1;
                *ss2 = 0;
                *rn = 1;
                return 5;
        }
        if (ss == (sublatticesize - 2)) {
                *ln = ss - 1;
                *ss2 = ss + 1;
                *rn = 0;          /* the right neighbor belongs to the “right” processor */
                return 6;
        }
        if (ss == (sublatticesize - 3)) {
                *ss2 = ss + 1;
                *rn = ss + 2;
                *ln = ss - 1;                     /* normal neighbor assignment, but locking is needed */
                return 7;
        }
}
 
/* return a random integer value that corresponds to an index in the sub-lattice arrays */
long random_at_most(long max)
{
       /* adapted from an online forum */ 
        unsigned long num_bins = (unsigned long) max + 1; 
        num_rand = (unsigned long) RAND_MAX + 1;
        bin_size = num_rand / num_bins;
        defect = num_rand % num_bins;
        long x;
        do {
                x = random();
        } while (num_rand - defect <= (unsigned long)x);
return x/bin_size;
}
 
/* sets all sites as down within a sub-lattice */
void init_array(int sublattice[],int sublatticesize)
{
        int taskid;
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid); /* grab processor’s identifier */
        int j = 0;
        for (j = 0; j < sublatticesize; j++) {
                sublattice[j] = DOWN;  /* set all sites as down */
        }
}
 
/* randomly distributes all of the initially-up sites within an initialized sub-lattice */
void randomization_init(int sublattice[], int * c, int * number_up, int * number_down, int sublatticesize)
{
        int max_index = sublatticesize - 1;
        int site = 0;
        while (*c > 0) {
        site = random_at_most(max_index);  /* select a random site to set as up */
                if (sublattice[site] == DOWN) {  /* make sure that site is down */
                	sublattice[site] = UP;   /* make the down site up */
*c = *c - 1;                         /* now one less site to up */
*number_up = *number_up + 1;       /* increase the amount of up sites by 1*/
     	            *number_down = sublatticesize - (*number_up);
    }
        }
}
 
/* function used for updating neighbors (C, D) when all sites are in the sub-lattice interior */
void update_near_neighbor(int sublattice[], int array[])
{         /* diagram of the four-site panel: [ln][rsn/ss][ss2][rn] or [C][A][B][D] */
        int ln = array[5];
        int ss = array[6];   
        int rn = array[7];
 
        if (sublattice[ss] == sublattice[ss + 1]) {
                sublattice[ln] = sublattice[ss];
                sublattice[rn] = sublattice[ss];     /* if UP1 is satisfied, the neighbors are updated */
        }
}
 
/* create file names for the data output files, open them for (appending) writing */
void create_and_open_files(FILE * output, FILE * output_MCS, FILE * output_m, char * latticesize, char * p, char * N)
{
        char * three_strings = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + 
	strlen("_Runs/") + strlen(p) + strlen(N) + 3 + strlen("_MCS_930.dat"));
        char * three_strings2 = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + 
	strlen("_Runs/") + strlen(p) + strlen(N) + 3 + strlen("_NC_930.dat"));
        char * three_strings3 = malloc(strlen("SM1_Data/N=") + 2*strlen(latticesize) + 
	strlen("_Runs/") + strlen(p) + strlen(N) + 3 + strlen("_m(t)_930.dat"));
       
        concatenate_strings(three_strings, "_MCS_930.dat", latticesize, p, N);
        concatenate_strings(three_strings2, "_NC_930.dat", latticesize, p, N);
        concatenate_strings(three_strings3, "_m(t)_930.dat", latticesize, p, N);
       
        output_MCS = fopen(three_strings, "a");   /*naming the MCS information file*/
        output = fopen(three_strings2, "a");      /*naming the E(x,L) information file.*/
        output_m = fopen(three_strings3, "a");
       
        free(three_strings);
        free(three_strings2);  /* free up the memory allocated for the string buffers */
        free(three_strings3);
}
 


/* called to create the strings for the file names used in create_and_open_files() */
void concatenate_strings(char * string, char * message, char * latticesize, char * p, char * N)
{
        strcpy(string, "SM1_Data/N=");
        strcat(string, latticesize);
        strcat(string, "_Runs/");
        strcat(string, latticesize);
        strcat(string, "_");                          /* create the file names by combining several strings */
        strcat(string, p);
        strcat(string, "_");
        strcat(string, N);
        strcat(string, message);
}
 
/* returns a random integer that ranges from 0 to P – 1 */
int return_rand_proc(int numtasks)
{
        int r;
        int min = 1;
        int max = numtasks;
        r = min + (rand() % (int)(max - min + 1));
        r -= 1;
        return r;
}
