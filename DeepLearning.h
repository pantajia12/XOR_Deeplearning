#include <stdio.h>
#include <malloc.h>
#include <math.h>

void reset();
double** forward();
void backward();
double** dot(double**, double**);
double** add(double**, double**);
double** subtract(double** a, double** b);
double** multiply(double** a, double** b);
double** broadcastingMultiply(double** a, double b);
double** sigmoid(double**);
double** softmax(double**);
double** relu(double**);
double** identity_function(double**);
double** transposedMatrix(double** x);
double** createArray(int, int);
double arrMax(double**);
double** createOne(int, int);

double** dot(double** x, double** W){//(x, W1)
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int n = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]
    int m = _msize(W[0])/sizeof(W[0][0])/(_msize(W)/sizeof(W[0])); //W[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){ 
            for(int k = 0; k < n; k++){
                result[i][j] += x[i][k] * W[k][j];//2차원 배열이 넘어오면서 1차원으로 변함
            }
        }
    }

    return result;
}

double** add(double** a, double** b){
    int x = _msize(a)/sizeof(a[0]); //a[n][]
    int y = _msize(b[0])/sizeof(b[0][0])/(_msize(b)/sizeof(b[0])); //b[][n]
    double **result = createArray(x, y);

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] + b[n][m];
        }
    }

    return result;
}

double** subtract(double** a, double** b){
    int x = _msize(a)/sizeof(a[0]); //a[n][]
    int y = _msize(b[0])/sizeof(b[0][0])/(_msize(b)/sizeof(b[0])); //b[][n]
    double **result = createArray(x, y);

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] - b[n][m];
        }
    }

    return result;
}

double** multiply(double** a, double** b){
    int x = _msize(a)/sizeof(a[0]); //a[n][]
    int y = _msize(b[0])/sizeof(b[0][0])/(_msize(b)/sizeof(b[0])); //b[][n]
    double **result = createArray(x, y);

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] * b[n][m];
        }
    }

    return result;
}

double** broadcastingMultiply(double** a, double b){ //파이썬 브로드캐스팅 기능
    int x = _msize(a)/sizeof(a[0]); //a[n][]
    int y = _msize(a[0])/sizeof(a[0][0])/(_msize(a)/sizeof(a[0])); //b[][n]
    double **result = createArray(x, y);

    for(int n = 0; n < x; n++){
        for(int m = 0; m < y; m++){
            result[n][m] = a[n][m] * b;
        }
    }

    return result;
}

double** sigmoid(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = 1 / (1 + pow(exp(1.0), (-x[i][j])));
        }
    }

    return result;
}

double** softmax(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    double Max = arrMax(x);
    double sumExpX = 0;

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            sumExpX += exp(x[i][j] - Max);
        }
    }

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = exp(x[i][j] - Max) / sumExpX;
        }
    }

    return result;
}

double** relu(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double **result = createArray(l, m);

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            result[i][j] = x[i][j] > 0 ? x[i][j] : 0;
        }
    }

    return result;
}

double** identity_function(double** x){
    return x;
}

double** transposedMatrix(double** x){ //전치행렬
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double** result = createArray(m, l);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < l; j++){
            result[i][j] = x[j][i];
        }
    }

    return result;
}

double** createArray(int l, int m){
    double **result; // l행 m열로 선언할거임

    result = calloc(sizeof(double*), l); 
    result[0] = calloc(sizeof(double), l * m);
    for (int i = 1; i < l; ++i) result[i] = result[i - 1] + m;

    return result;
}

double arrMax(double** x){
    int l = _msize(x)/sizeof(x[0]); //x[n][]
    int m = _msize(x[0])/sizeof(x[0][0])/(_msize(x)/sizeof(x[0])); //x[][n]

    double max = x[0][0];

    for(int i = 0; i < l; i++){
        for(int j = 0; j < m; j++){
            if(max < x[i][j]){
                max = x[i][j];
            }
        }
    }

    return max;
}

double** createOne(int x, int y){ //시그모이드 오차역전파에 사용
    double** result = createArray(x, y);

    for(int i = 0; i < x; i++){
        for(int j = 0; j < y; j++){
            result[i][j] = 1;
        }
    }

    return result;
}