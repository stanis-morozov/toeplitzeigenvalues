#include "mpi.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <queue>
#include <cmath>
#include <random>

//#define PRINTDEBUG

const int INTERVAL_FINISHED = 1;
const int EIGENVALUE_FINISHED = 2;
const int CONTINUE = 3;
const int END = 4;
const double EPS = 1e-15;
const char *output = "result.txt";
const char *timefile = "time.txt";


struct Interval
{
    double l, r;
    int p, q;
    Interval(double l_, double r_, int p_, int q_)
    {
        l = l_;
        r = r_;
        p = p_;
        q = q_;
    }
};


std::pair <int, double> neg(double lambda, int n, double *t, double *z, double *zp, double *q)
{
    q[1] = t[0] - lambda;
    zp[1] = t[1] / q[1];

    for (int m = 2; m <= n; m++) {
        q[m] = (1 - zp[m - 1] * zp[m - 1]) * q[m - 1];
        double tmp = 0;
        for (int i = 1; i <= m - 1; i++) {
            tmp += t[m - i] * zp[i];
        }
        z[m] = 1 / q[m] * (t[m] - tmp);
        for (int j = 1; j <= m - 1; j++) {
            z[j] = zp[j] - z[m] * zp[m - j];
        }
        for (int j = 1; j <= m; j++) {
            zp[j] = z[j];
        }
    }
    int cnt = 0;
    for (int i = 1; i <= n; i++) {
        if (q[i] < 0) {
            cnt++;
        }
    }
    return std::make_pair(cnt, q[n]);
}

int main(int argc, char **argv)
{
    int rc;
    int numproc;
    int myrank;
    int n;
    int p, q;
    double *t, *z, *zp, *delta;
    double t1, t2;

    if (rc = MPI_Init(&argc, &argv)) {
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


    const int nitems = 4;
    int blocklength[4] = {1, 1, 1, 1};
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT};
    MPI_Datatype MPI_INTERVAL;
    MPI_Aint offsets[4];
    offsets[0] = offsetof(Interval, l);
    offsets[1] = offsetof(Interval, r);
    offsets[2] = offsetof(Interval, p);
    offsets[3] = offsetof(Interval, q);
    MPI_Type_create_struct(nitems, blocklength, offsets, types, &MPI_INTERVAL);
    MPI_Type_commit(&MPI_INTERVAL);



    if (myrank == 0) {
        if (argc > 1) {
            sscanf(argv[1], "%d", &n);
        } else {
            n = 10 * 1000;
        }
        t = (double*)malloc((n + 1) * sizeof(*t));
        for (int i = 0; i < n; i++) {
            if (i == 0) {
                t[i] = M_PI * M_PI / 3;
            } else {
                t[i] = (2 * i * M_PI * cos(i * M_PI) + (i * i * M_PI * M_PI - 2) * sin(i * M_PI)) / (i * i * i * M_PI);
            }
        }
        t1 = MPI_Wtime();
        for (int i = 1; i < numproc; i++) {
            MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(t, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            int tmp[2] = {(i - 1) * n / (numproc - 1) + 1, i * n / (numproc - 1)};
            MPI_Send(tmp, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t = (double*)malloc((n + 1) * sizeof(*t));
        MPI_Recv(t, n, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int tmp[2];
        MPI_Recv(tmp, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p = tmp[0];
        q = tmp[1];
    }

    if (myrank == 0) {
        int *workers = (int*)malloc(numproc * sizeof(*workers));
        std::queue<int> freew;
        std::fill(workers, workers + numproc, -1);
        std::queue<Interval> extracted;
        Interval *tmp = (Interval*)malloc((n + 1) * sizeof(*tmp));
        double *result = (double*)malloc((n + 1) * sizeof(*result));
        int computed = 0;
        int type;

        int ttt = 0;

        while (computed < n) {
#ifdef PRINTDEBUG
            std::cout << "waiting..." << std::endl;
#endif
            MPI_Status status;
            MPI_Recv(&type, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
#ifdef PRINTDEBUG
            std::cout << "Message from " << status.MPI_SOURCE << std::endl;
#endif

            if (type == INTERVAL_FINISHED) {
#ifdef PRINTDEBUG
                std::cout << "Intervals computed by " << status.MPI_SOURCE << std::endl;
#endif
                int i = status.MPI_SOURCE;
                workers[i] = 0;
                freew.push(i);
                int k = -((i - 1) * n / (numproc - 1) + 1) + (i * n / (numproc - 1)) + 1;
                MPI_Recv(tmp, k, MPI_INTERVAL, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i < k; i++) {
                    extracted.push(tmp[i]);
                    ttt++;
                }
            } else if (type == EIGENVALUE_FINISHED) {
#ifdef PRINTDEBUG
                std::cout << "Eigenvalue computed by " << status.MPI_SOURCE << std::endl;
#endif
                MPI_Recv(&result[workers[status.MPI_SOURCE]], 1, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                workers[status.MPI_SOURCE] = 0;
                freew.push(status.MPI_SOURCE);
                computed++;
            }

            while (!extracted.empty() && !freew.empty()) {
                int w = freew.front();
                freew.pop();
                Interval curr = extracted.front();
                extracted.pop();
                workers[w] = curr.p;
                int type = CONTINUE;
                MPI_Send(&type, 1, MPI_INT, w, myrank, MPI_COMM_WORLD);
                MPI_Send(&curr, 1, MPI_INTERVAL, w, myrank, MPI_COMM_WORLD);
            }
            
        }

        for (int i = 1; i < numproc; i++) {
            int type = END;
            MPI_Send(&type, 1, MPI_INT, i, myrank, MPI_COMM_WORLD);
        }
        t2 = MPI_Wtime();
        std::cout << t2 - t1 << " s" << std::endl;
        for (int i = 1; i <= n; i++) {
            std::cout << std::setprecision(40) << std::fixed << result[i] << std::endl;
        }
        free(tmp);
        free(workers);
        free(result);
    } else {
        z = (double*)malloc((n + 1) * sizeof(*z));
        zp = (double*)malloc((n + 1) * sizeof(*zp));
        delta = (double*)malloc((n + 1) * sizeof(*delta));
        double mx = 0;
        double s1, s2;
        for (int i = 0; i < n; i++) {
            s1 = s2 = 0;
            for (int j = 1; j <= i; j++) {
                s1 += fabs(t[j]);
            }
            for (int j = 1; j < n - i; j++) {
                s2 += fabs(t[j]);
            }
            mx = std::max(mx, s1 + s2);
        }
        srand(time(0));
        s1 = (double)rand() / RAND_MAX;
        s2 = (double)rand() / RAND_MAX;

        std::queue <Interval> intervals;
        intervals.push(Interval(t[0] - mx - s1, t[0] + mx + s2, p, q));
        std::vector <Interval> extracted;

        while (!intervals.empty()) {
            Interval curr = intervals.front();
            intervals.pop();
        
            if (curr.q == curr.p) {
                std::pair <int, double> pra, prb;
                pra = neg(curr.l, n, t, z, zp, delta);
                prb = neg(curr.r, n, t, z, zp, delta);
                if (pra.first == curr.p - 1 && prb.first == curr.p) {
                    if (pra.second > 0 && prb.second < 0) {
                        extracted.push_back(curr);
                        continue;
                    }
                }
            }
        
            double lambda = (curr.l + curr.r) / 2;
            int ng = neg(lambda, n, t, z, zp, delta).first;
            if (ng <= curr.p - 1) {
                intervals.push(Interval(lambda, curr.r, curr.p, curr.q));
            }
            if (ng >= curr.p) {
                if (ng >= curr.q) {
                    intervals.push(Interval(curr.l, lambda, curr.p, curr.q));
                } else {
                    intervals.push(Interval(curr.l, lambda, curr.p, ng));
                    intervals.push(Interval(lambda, curr.r, ng + 1, curr.q));
                }
            }
        }

        int type = INTERVAL_FINISHED;
        MPI_Send(&type, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD);
        MPI_Send(extracted.data(), (int)extracted.size(), MPI_INTERVAL, 0, myrank, MPI_COMM_WORLD);

        while (1) {
            MPI_Recv(&type, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (type == END) {
                break;
            } else if (type == CONTINUE) {
                Interval curr(0, 0, 0, 0);
                MPI_Recv(&curr, 1, MPI_INTERVAL, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                double a = curr.l, b = curr.r;
                int p = curr.p;
                while (b - a > EPS) {
                    double lambda = (a + b) / 2;
                    int ng = neg(lambda, n, t, z, zp, delta).first;
                    if (ng <= p - 1) {
                        a = lambda;
                    }
                    if (ng >= p) {
                        b = lambda;
                    }
                }
                double result = (a + b) / 2;
                type = EIGENVALUE_FINISHED;
                MPI_Send(&type, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD);
                MPI_Send(&result, 1, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD);
            }
        }
    }

    free(t);
    if (myrank != 0) {
        free(z);
        free(zp);
        free(delta);
    }

    MPI_Type_free(&MPI_INTERVAL);
    MPI_Finalize();
    return 0;
}
