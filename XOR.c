/*
 x1 =(W1)=> a1(b1)
    \      /      \
      \  /(W3)     \(W5)
      /\            y(b3)
     /  (W2)       /(W6)
    /      \      /
 x2  =(W4)=> a2(b2)

*/
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include "DeepLearning.h"

void test();
double** MSE();

double** X; //입력 초기화

double** x1; //입력 1
double** x2; //입력 2

double** W1;
double** W2;
double** W3;
double** W4;
double** W5;
double** W6;

double** b1;
double** b2;
double** b3;

double** a1;
double** a2;

double** Y; //정답

double** OUT; //나의 답

int n;

void reset(){
    X = createArray(4, 2);
    Y = createArray(4, 1);

    x1 = createArray(1, 1);
    x2 = createArray(1, 1);

    W1 = createArray(1, 1);
    W2 = createArray(1, 1);
    W3 = createArray(1, 1);
    W4 = createArray(1, 1);
    W5 = createArray(1, 1);
    W6 = createArray(1, 1);

    b1 = createArray(1, 1);
    b2 = createArray(1, 1);
    b3 = createArray(1, 1);

    a1 = createArray(1, 1);
    a2 = createArray(1, 1);

    X[0][0] = 0, X[0][1] = 0;
    X[1][0] = 0, X[1][1] = 1;
    X[2][0] = 1, X[2][1] = 0;
    X[3][0] = 1, X[3][1] = 1;

    Y[0][0] = 0;
    Y[1][0] = 1;
    Y[2][0] = 1;
    Y[3][0] = 0;

    srand(time(NULL));
    x1[0][0] = rand()%3 + 1;
    x2[0][0] = rand()%3 + 1;
    W1[0][0] = rand()%3 + 1;
    W2[0][0] = rand()%3 + 1;
    W3[0][0] = rand()%3 + 1;
    W4[0][0] = rand()%3 + 1;
    W5[0][0] = rand()%3 + 1;
    W6[0][0] = rand()%3 + 1;
    b1[0][0] = rand()%3 + 1;
    b2[0][0] = rand()%3 + 1;
    b3[0][0] = rand()%3 + 1;

}

int main(){
    double** error = createArray(1, 1);
    error[0][0] = 100;
    for(;error[0][0] > 0.00005;){//오류가 일정 수치 미만까지 도달할 때 까지 반복
        reset();
        for(int epoch = 0; epoch < 100000; epoch++){
            n = epoch % 4;
            x1[0][0] = X[n][0], x2[0][0] = X[n][1];

            OUT = forward();
            backward();
        }
        
        error = MSE();
    }
    test();
    puts("----------------------");
    printf("error: %lf\n", error[0][0]);
    printf("W1: %lf\n", W1[0][0]);
    printf("W2: %lf\n", W2[0][0]);
    printf("W3: %lf\n", W3[0][0]);
    printf("W4: %lf\n", W4[0][0]);
    printf("W5: %lf\n", W5[0][0]);
    printf("W6: %lf\n", W6[0][0]);
    printf("b1: %lf\n", b1[0][0]);
    printf("b2: %lf\n", b2[0][0]);
    printf("b3: %lf\n", b3[0][0]);

}

void test(){
    puts("x1  x2   y");
    for(int epoch = 0; epoch < 4; epoch++){
        n = epoch % 4;
        x1[0][0] = X[n][0], x2[0][0] = X[n][1];
        OUT = forward();
        printf("%.0lf   %.0lf    %lf\n", x1[0][0], x2[0][0], OUT[0][0]);
    }
}

double** forward(){
    a1 = add(add(multiply(x1, W1), multiply(x2, W3)), b1);
    a1 = sigmoid(a1);

    a2 = add(add(multiply(x1, W2), multiply(x2, W4)), b2);
    a2 = sigmoid(a2);

    OUT = add(add(multiply(a1, W5), multiply(a2, W6)), b3);
    OUT = sigmoid(OUT);

    return OUT;
}

void backward(){
    double** error = createArray(1, 1);
    error[0][0] = OUT[0][0] - Y[n][0];

    double** one = createOne(1, 1);
    double** dOUT = multiply(error, multiply(OUT, subtract(one, OUT)));
    double** db3 = dOUT;
    double** dW5 = multiply(dOUT, a1);
    double** dW6 = multiply(dOUT, a2);

    double** da1Sigmoid = multiply(dOUT, W5);
    double** da2Sigmoid = multiply(dOUT, W6);

    double** da1 = multiply(da1Sigmoid, multiply(a1, subtract(one, a1)));
    double** da2 = multiply(da2Sigmoid, multiply(a2, subtract(one, a2)));

    double** db1 = da1;
    double** dW1 = multiply(da1, x1);
    double** dW3 = multiply(da1, x2);

    double** db2 = da2;
    double** dW2 = multiply(da2, x1);
    double** dW4 = multiply(da2, x2);

    double learning_rate = 1; //학습률

    W1 = subtract(W1, broadcastingMultiply(dW1, learning_rate));
    W2 = subtract(W2, broadcastingMultiply(dW2, learning_rate));
    W3 = subtract(W3, broadcastingMultiply(dW3, learning_rate));
    W4 = subtract(W4, broadcastingMultiply(dW4, learning_rate));
    W5 = subtract(W5, broadcastingMultiply(dW5, learning_rate));
    W6 = subtract(W6, broadcastingMultiply(dW6, learning_rate));
    b1 = subtract(b1, broadcastingMultiply(db1, learning_rate));
    b2 = subtract(b2, broadcastingMultiply(db2, learning_rate));
    b3 = subtract(b3, broadcastingMultiply(db3, learning_rate));
}

double** MSE(){//평균 제곱 오차 0: 오차 없음 1 도달할수록 오차 커짐
    double** result = createArray(1, 1);
    result[0][0] = 0;
    for(int i = 0; i < 4; i++){
        result = add(result, multiply(subtract(OUT, Y), subtract(OUT, Y)));
    }
    result = broadcastingMultiply(result, 0.25);
    return result;
}