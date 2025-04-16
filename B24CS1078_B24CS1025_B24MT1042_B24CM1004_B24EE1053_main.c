#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "B24CS1078_B24CS1025_B24MT1042_B24CM1004_B24EE1053_constants.h"
#include <allheaders.h>

float *float_calloc(int size)
{
    float *temp = (float *)calloc(size, sizeof(float));
    if (!temp)
    {
        printf("Memory allocation failed!!");
        exit(1);
    }
    return temp;
}

float ReLU(float x)
{//RELU=rectifed linear unit
    if (x >= 0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

float ReLU_derivative(float x)
{
    if (x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void softmax(int size, float *input, float *result)
{
    float max = input[0];
    for (int i = 1; i < size; i++)
    {
        if (input[i] > max)
            max = input[i];
    }

    float sum = 0;
    for (int i = 0; i < size; i++)
    {
        result[i] = exp(input[i] - max); // Prevents overflow
        sum += result[i];
    }

    for (int i = 0; i < size; i++)
    {
        result[i] /= sum;
    }
}

float random_float()
{
    return ((float)rand() / (float)RAND_MAX);// genrate random no b/t 0 to 1
}

void matrix_multiplication(int row1, int col1, int row2, int col2, float *matrix1, float *matrix2, float *result)
/*
Here col1 & row2 are equal otherwise multiplication is not possible
(so result matrix will be of size (row1 x col2))
Function has been verified with test cases
*/
{
    for (int k = 0; k < row1; k++) // moves in row of result matrix
    {
        for (int j = 0; j < col2; j++) // moves in column of result matrix
        {
            *result = 0;
            for (int i = 0; i < row2; i++) // for each element in result
            {
                *result += (*matrix1) * (*matrix2);
                matrix1++;
                matrix2 += col2;
            }
            matrix2 -= (row2 * col2);
            matrix1 -= col1;
            matrix2++;
            result++;
        }
        matrix2 -= col2;
        matrix1 += col1;
    }
}

void matrix_transpose(int row, int col, float *matrix, float *result)
/*
Function has been verified with test cases
*/
{
    for (int j = 0; j < col; j++)
    {
        for (int i = 0; i < row; i++)
        {
            *result = *matrix;
            result++;
            matrix += col;
        }
        matrix -= (row * col);
        matrix++;
    }
}

void matrix_addition(int row, int col, float *matrix1, float *matrix2, float *result)
/*
Function has been verified with test cases
*/
{
    for (int i = 0; i < (row * col); i++)
    {
        *result = *matrix1 + *matrix2;
        matrix1++;
        matrix2++;
        result++;
    }
}

void matrix_substraction(int row, int col, float *matrix1, float *matrix2, float *result)
/*
Function has been verified with test cases
*/
{
    for (int i = 0; i < (row * col); i++)
    {
        *result = *matrix1 - *matrix2;
        matrix1++;
        matrix2++;
        result++;
    }
}

void weights_initialization(int row, int col, float *matrix)
{//to give random values in matrix from 0 to 1
    for (int i = 0; i < (row * col); i++)
    {
        matrix[i] = random_float() * 0.1f;
    }
}

void mnist_image_loader(const char *file, int size, unsigned char *image_input_matrix, int index)
{//load image  from training dataset file
    FILE *file_pointer = fopen(file, "rb");
    if (file_pointer == NULL)
    {
        printf("Error opening file.\n");
        exit(1);
    }
    fseek(file_pointer, 16 + (index * size), SEEK_SET);// first 16 byte are useless

    fread(image_input_matrix, sizeof(unsigned char), size, file_pointer);

    fclose(file_pointer);
    return;
}

void mnist_label_loader(const char *file, int size, unsigned char *image_input_matrix, int index)
{// load label from training dataset file
    FILE *file_pointer = fopen(file, "rb");
    if (file_pointer == NULL)
    {
        printf("Error opening file.\n");
        exit(1);
    }
    fseek(file_pointer, 8 + (index * size), SEEK_SET);// first 8 byte are useless

    fread(image_input_matrix, 1, 1, file_pointer);

    fclose(file_pointer);
    return;
}
int forward_propogation(float *input, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, float *output)
{
    /*
    x=784x1
    w1=512x784
    b1=512x1
    w2=128x512
    b2=128x1
    w3=10x128
    b3=10x1

    input=x
    z1=w1x+b1
    A1=relu(w1x+b1)
    z2=W2A1+b2
    A2=relu(w2A1+b2)
    z3=w3A2+b3
    softmax(W3A2+b3)
    return maximum of (softmax(W3A2+b3));
    */
    // Calculating A1
    float *x1 = float_calloc(h1_size);
    matrix_multiplication(h1_size, image_size, image_size, 1, w1, input, x1); // training ke time kaam karega
    float *z1 = float_calloc(h1_size);
    matrix_addition(h1_size, 1, x1, b1, z1);
    float *a1 = float_calloc(h1_size);
    for (int i = 0; i < h1_size; i++)
    {
        *(a1 + i) = ReLU(*(z1 + i));
    }

    // Calculating A2
    float *x2 = float_calloc(h2_size);
    matrix_multiplication(h2_size, h1_size, h1_size, 1, w2, a1, x2);
    float *z2 = float_calloc(h2_size);
    matrix_addition(h2_size, 1, x2, b2, z2);
    float *a2 = float_calloc(h2_size);
    for (int i = 0; i < h2_size; i++)
    {
        *(a2 + i) = ReLU(*(z2 + i));
    }

    // Calculating softmax
    float *x3 = float_calloc(output_numbers);
    matrix_multiplication(output_numbers, h2_size, h2_size, 1, w3, a2, x3);
    float *z3 = float_calloc(output_numbers);
    matrix_addition(output_numbers, 1, x3, b3, z3);
    softmax(output_numbers, z3, output);
    int idx = 0;
    float max = output[0];
    for (int i = 0; i < output_numbers; i++)
    {
        if (output[i] > output[idx])
        {
            idx = i;
        }
    }
    free(x1);
    free(x2);
    free(x3);
    free(z1);
    free(z2);
    free(z3);
    free(a1);
    free(a2);
    return idx;
}

void weights_update(float *input, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, int true_label)
{
    /*
    z1=w1x+b1
    A1=relu(w1x+b1)
    z2=W2A1+b2
    A2=relu(w2A1+b2)
    z3=w3A2+b3
    y is hot vector
    ex if output of forward_propogation function is 7
    then y will be [0,0,0,0,0,0,0,1,0,0]
    */
    // Calculating A1
    float *x1 = float_calloc(h1_size);
    matrix_multiplication(h1_size, image_size, image_size, 1, w1, input, x1); // weights me trainig ke time update karne ke liye
    float *z1 = float_calloc(h1_size);
    matrix_addition(h1_size, 1, x1, b1, z1);
    float *a1 = float_calloc(h1_size);
    for (int i = 0; i < h1_size; i++)
    {
        *(a1 + i) = ReLU(*(z1 + i));
    }

    // Calculating A2
    float *x2 = float_calloc(h2_size);
    matrix_multiplication(h2_size, h1_size, h1_size, 1, w2, a1, x2);
    float *z2 = float_calloc(h2_size);
    matrix_addition(h2_size, 1, x2, b2, z2);
    float *a2 = float_calloc(h2_size);
    for (int i = 0; i < h2_size; i++)
    {
        *(a2 + i) = ReLU(*(z2 + i));
    }

    // Calculating softmax
    float *x3 = float_calloc(output_numbers);
    matrix_multiplication(output_numbers, h2_size, h2_size, 1, w3, a2, x3);
    float *z3 = float_calloc(output_numbers);
    matrix_addition(output_numbers, 1, x3, b3, z3);
    float *output_matrix = float_calloc(output_numbers);
    softmax(output_numbers, z3, output_matrix);
    float max = 0.0;
    int idx = 0;
    for (int i = 0; i < output_numbers; i++)
    {
        if (max < *(output_matrix + i))
        {
            max = *(output_matrix + i);
            idx = i;
        }
    }

    // Hot Vector
    float *y = float_calloc(output_numbers);
    *(y + true_label) = 1;

    // Backward Propagation

    // Output Layer
    float *dz3 = float_calloc(output_numbers);
    matrix_substraction(output_numbers, 1, output_matrix, y, dz3);
    float *db3 = float_calloc(output_numbers);
    for (int i = 0; i < output_numbers; i++)
    {
        *(db3 + i) = *(dz3 + i);
    }
    float *a2t = float_calloc(h2_size);
    matrix_transpose(h2_size, 1, a2, a2t);
    float *dw3 = float_calloc(output_numbers * h2_size);
    matrix_multiplication(output_numbers, 1, 1, h2_size, dz3, a2t, dw3);

    // Second Hidden Layer
    float *w3t = float_calloc(output_numbers * h2_size);
    matrix_transpose(output_numbers, h2_size, w3, w3t);
    float *da2 = float_calloc(h2_size);
    matrix_multiplication(h2_size, output_numbers, output_numbers, 1, w3t, dz3, da2);
    float *dz2 = float_calloc(h2_size);
    float *rz2 = float_calloc(h2_size);
    for (int i = 0; i < h2_size; i++)
    {
        *(rz2 + i) = ReLU_derivative(*(z2 + i));
        *(dz2 + i) = *(da2 + i) * (*(rz2 + i));
    }
    float *a1t = float_calloc(h1_size);
    matrix_transpose(h1_size, 1, a1, a1t);
    float *dw2 = float_calloc(h2_size * h1_size);
    matrix_multiplication(h2_size, 1, 1, h1_size, dz2, a1t, dw2);
    float *db2 = float_calloc(h2_size);
    for (int i = 0; i < h2_size; i++)
    {
        *(db2 + i) = *(dz2 + i);
    }

    // First Hidden Layer
    float *w2t = float_calloc(h1_size * h2_size);
    matrix_transpose(h2_size, h1_size, w2, w2t);
    ;
    float *da1 = float_calloc(h1_size);
    matrix_multiplication(h1_size, h2_size, h2_size, 1, w2t, dz2, da1);
    float *dz1 = float_calloc(h1_size);
    float *rz1 = float_calloc(h1_size);
    for (int i = 0; i < h1_size; i++)
    {
        *(rz1 + i) = ReLU_derivative(*(z1 + i));
        *(dz1 + i) = *(da1 + i) * (*(rz1 + i));
    }
    float *inpt = float_calloc(image_size);
    matrix_transpose(image_size, 1, input, inpt);
    float *dw1 = float_calloc(h1_size * image_size);
    matrix_multiplication(h1_size, 1, 1, image_size, dz1, inpt, dw1);
    float *db1 = float_calloc(h1_size);
    for (int i = 0; i < h1_size; i++)
    {
        *(db1 + i) = *(dz1 + i);
    }

    // Updating W1
    for (int i = 0; i < h1_size * image_size; i++)
    {
        float temp = *(dw1 + i);
        *(w1 + i) -= learning_rate * temp;
    }

    // Updating W2
    for (int i = 0; i < h2_size * h1_size; i++)
    {
        float temp = *(dw2 + i);
        *(w2 + i) -= learning_rate * temp;
    }

    // Updating W3
    for (int i = 0; i < output_numbers * h2_size; i++)
    {
        float temp = *(dw3 + i);
        *(w3 + i) -= learning_rate * temp;
    }

    // Updating B1
    for (int i = 0; i < h1_size; i++)
    {
        float temp = *(db1 + i);
        *(b1 + i) -= learning_rate * temp;
    }

    // Updating B2
    for (int i = 0; i < h2_size; i++)
    {
        float temp = *(db2 + i);
        *(b2 + i) -= learning_rate * temp;
    }

    // Updating B3
    for (int i = 0; i < output_numbers; i++)
    {
        float temp = *(db3 + i);
        *(b3 + i) -= learning_rate * temp;
    }
    free(x1);
    free(z1);
    free(a1);
    free(x2);
    free(z2);
    free(a2);
    free(x3);
    free(z3);
    free(dz3);
    free(db3);
    free(a2t);
    free(dw3);
    free(w3t);
    free(da2);
    free(dz2);
    free(rz2);
    free(a1t);
    free(dw2);
    free(db2);
    free(w2t);
    free(da1);
    free(dz1);
    free(rz1);
    free(inpt);
    free(dw1);
    free(db1);
    free(y);
}

void print_confusion_matrix(int *a)
{//matrix ke diagonal me correct guess print hoge baki element me wrong guess ayege
    printf("Confusion Matrix:\n");
    printf("ROWS are predicted and COLUMNS are actual\n\n");

    printf("     ");
    for (int i = 0; i < 10; i++)
    {
        printf("%4d", i);
    }
    printf("\n     ");
    for (int i = 0; i < 10; i++)
    {
        printf("----");
    }
    printf("\n");

    for (int i = 0; i < 10; i++)
    {
        printf("%3d |", i);
        for (int j = 0; j < 10; j++)
        {
            printf("%4d", *(a + i * 10 + j));
        }
        printf("\n");
    }
}

void load_weights_bias(const char *file, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3)
{//pehle se trained weight and bias vali file load karne ke liye
    FILE *fptr = fopen(file, "rb");
    if (fptr == NULL)
    {
        printf("Error loading weights and bias!!");
        exit(1);
    }
    fread(w1, sizeof(float), (h1_size * image_size), fptr);
    fread(b1, sizeof(float), (h1_size * 1), fptr);
    fread(w2, sizeof(float), (h2_size * h1_size), fptr);
    fread(b2, sizeof(float), (h2_size * 1), fptr);
    fread(w3, sizeof(float), (output_numbers * h2_size), fptr);
    fread(b3, sizeof(float), (output_numbers * 1), fptr);
    fclose(fptr);
    printf("Weights and Bias loaded successfully!!\n");
    return;
}

void save_weight_bias(const char *file, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3)
{// weight and bias ko training ke baad save karne ke liye
    FILE *fptr = fopen(file, "wb");
    if (fptr == NULL)
    {
        printf("Error loading weights and bias!!");
        exit(1);
    }
    fwrite(w1, sizeof(float), (h1_size * image_size), fptr);
    fwrite(b1, sizeof(float), (h1_size * 1), fptr);
    fwrite(w2, sizeof(float), (h2_size * h1_size), fptr);
    fwrite(b2, sizeof(float), (h2_size * 1), fptr);
    fwrite(w3, sizeof(float), (output_numbers * h2_size), fptr);
    fwrite(b3, sizeof(float), (output_numbers * 1), fptr);
    fclose(fptr);
    printf("Saved weights and bias!!\n");
    return;
}

float *user_given_image(const char *filename)
{
    PIX *imgptr = pixRead(filename);
    PIX *gray = pixConvertTo8(imgptr, 0);
    PIX *resized_imgptr = pixScaleToSize(gray, 28, 28);
    float *input_matrix = (float *)calloc(784, sizeof(float));
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            unsigned int pixel;
            pixGetPixel(resized_imgptr, j, i, &pixel);
            *(input_matrix + i * 28 + j) =((float)pixel / 255.0f);
        }
    }
    printf("\n");
    pixDestroy(&resized_imgptr);
    pixDestroy(&imgptr);
    pixDestroy(&gray);
    return input_matrix;
}

int main()
{
    srand(time(NULL));
    float w1[h1_size][image_size];
    float b1[h1_size][1] = {0};
    float w2[h2_size][h1_size];//intialization weight and bias
    float b2[h2_size][1] = {0};
    float w3[output_numbers][h2_size];
    float b3[output_numbers][1] = {0};

    float *ptr_w1 = &w1[0][0];
    float *ptr_b1 = &b1[0][0];
    float *ptr_w2 = &w2[0][0];
    float *ptr_b2 = &b2[0][0];// pointers for weight and bias 
    float *ptr_w3 = &w3[0][0];
    float *ptr_b3 = &b3[0][0];

    weights_initialization(h1_size, image_size, ptr_w1);
    weights_initialization(h2_size, h1_size, ptr_w2);//weights initialization 
    weights_initialization(output_numbers, h2_size, ptr_w3);

    const char *file = "train-images.idx3-ubyte";
    const char *file_label = "train-labels.idx1-ubyte";//training dataset files

    const char *file2 = "t10k-images.idx3-ubyte";
    const char *file2_label = "t10k-labels.idx1-ubyte";//testing dataset files

    printf("1) Train and Test\n");//train model for n epochs,test and print confusion matrix and accuracy
    printf("2) Test from already trained model\n");//test from trained model(using saved weights and bias files
    printf("3) Update trained model\n");// accurate saved weights and bias file
    printf("4) Test from user given MNIST demo image\n");
    int n;
    printf("Enter your choice:-");
    scanf("%d", &n);
    switch (n)
    {
    case 1:
    {
        int EPOCH;
        printf("Enter Epoches:-");
        scanf("%d", &EPOCH);//iteration for training from mnist dataset(60000)
        printf("Enter how often you want to print the outcome:-");// kitne images ke baad accuracy print karni he
        int often=0;
        scanf("%d", &often);

        for (int e = 0; e < EPOCH; e++)
        {//for each epoch train from mnist dataset
            int correct = 0;
            int Accuracy_temp=0;
            for (int current_images = 0; current_images < train_images; current_images++)
            {
                unsigned char *image_input_matrix = (unsigned char *)calloc(784, sizeof(unsigned char));
                unsigned char *label_input_matrix = (unsigned char *)calloc(1, sizeof(unsigned char));

                mnist_image_loader(file, 784, image_input_matrix, current_images);
                mnist_label_loader(file_label, 1, label_input_matrix, current_images);//loading image from dataset

                float *ptr_output = (float *)calloc(10, sizeof(float));

                int input_label = *label_input_matrix;//actual output no from 0 to 9
                float input[image_size][1];

                for (int i = 0; i < image_size; i++)
                {
                    input[i][0] = (float)((*(image_input_matrix + i)) / 255.0f);
                }
                float *ptr_input = &input[0][0];
                int guess = forward_propogation(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);//return the number predicted by model

                weights_update(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, input_label);//performs backpropagation to update weights and bias
                if (current_images % often == 0)
                {
                    printf("Predicted: %d | Actual: %d\n", guess, input_label); //prints the predicted and actual number
                    printf("Accuracy for last %d processed images: %.2f\n",often,(float)Accuracy_temp/often* 100.0);
                    Accuracy_temp=0;
                }
                if (guess == input_label)
                {
                    Accuracy_temp++;
                    correct++;

                }
                free(image_input_matrix);
                free(label_input_matrix);
                free(ptr_output);

                if ((current_images + 1) % often == 0)
                {
                    printf("Epoch %d | Images Processed: %d | Accuracy: %.2f%%\n", e + 1, current_images + 1, ((float)correct / (current_images + 1)) * 100.0);
                    printf("\n");
                }

            }
            printf("Epoch %d accuracy: %f\n", e + 1, ((float)correct / train_images) * 100.0);
        }
        //test from test data set 
        int correct = 0;
        int confusion_matrix[10][10] = {0};
        int *confusion_matrix_ptr = &confusion_matrix[0][0];
        for (int current_images = 0; current_images < test_images; current_images++)
        {
            unsigned char *image_input_matrix = (unsigned char *)calloc(784, sizeof(unsigned char));
            unsigned char *label_input_matrix = (unsigned char *)calloc(1, sizeof(unsigned char));

            mnist_image_loader(file2, 784, image_input_matrix, current_images);
            mnist_label_loader(file2_label, 1, label_input_matrix, current_images);//load files from testing data set

            float *ptr_output = (float *)calloc(10, sizeof(float));

            int input_label = *label_input_matrix;
            float input[image_size][1];

            for (int i = 0; i < image_size; i++)
            {
                input[i][0] = (float)((*(image_input_matrix + i)) / 255.0f);
            }
            float *ptr_input = &input[0][0];
            int guess = forward_propogation(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);
            confusion_matrix[guess][input_label]++;
            if (guess == input_label)
            {
                correct++;
            }
            free(image_input_matrix);
            free(label_input_matrix);
            free(ptr_output);
        }
        printf("Testing accuracy: %f\n", ((float)correct / test_images) * 100.0);//testing accuracy
        print_confusion_matrix(confusion_matrix_ptr);

        break;
    }
    
    case 2:
    {
        load_weights_bias("B24CS1078_B24CS1025_B24MT1042_B24CM1004_B24EE1053_weight_bias.bin", ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3);//load saved weights and bias 
        int correct = 0;
        int confusion_matrix[10][10] = {0};
        int *confusion_matrix_ptr = &confusion_matrix[0][0];
        for (int current_images = 0; current_images < test_images; current_images++)
        {
            unsigned char *image_input_matrix = (unsigned char *)calloc(784, sizeof(unsigned char));
            unsigned char *label_input_matrix = (unsigned char *)calloc(1, sizeof(unsigned char));

            mnist_image_loader(file2, 784, image_input_matrix, current_images);
            mnist_label_loader(file2_label, 1, label_input_matrix, current_images);//load testing data set from mnist

            float *ptr_output = (float *)calloc(10, sizeof(float));

            int input_label = *label_input_matrix;
            float input[image_size][1];

            for (int i = 0; i < image_size; i++)
            {
                input[i][0] = (float)((*(image_input_matrix + i)) / 255.0f);//convert each input into 0 to 1 range
            }
            float *ptr_input = &input[0][0];
            int guess = forward_propogation(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);//guess no from trained model
            confusion_matrix[guess][input_label]++;
            if (guess == input_label)
            {
                correct++;
            }
            free(image_input_matrix);
            free(label_input_matrix);
            free(ptr_output);
        }
        printf("Testing accuracy: %f\n", ((float)correct / test_images) * 100.0);
        print_confusion_matrix(confusion_matrix_ptr);
    
        break;
    }
    
    case 3:
    {
        load_weights_bias("B24CS1078_B24CS1025_B24MT1042_B24CM1004_B24EE1053_weight_bias.bin", ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3);//load saved weight and bias 

        int EPOCH;
        printf("Enter Epoches:-");//epoch for more training of our saved weight and bias 
        scanf("%d", &EPOCH);

        for (int e = 0; e < EPOCH; e++)
        {
            int correct = 0;
            for (int current_images = 0; current_images < train_images; current_images++)
            {
                unsigned char *image_input_matrix = (unsigned char *)calloc(784, sizeof(unsigned char));
                unsigned char *label_input_matrix = (unsigned char *)calloc(1, sizeof(unsigned char));

                mnist_image_loader(file, 784, image_input_matrix, current_images);
                mnist_label_loader(file_label, 1, label_input_matrix, current_images);

                float *ptr_output = (float *)calloc(10, sizeof(float));

                int input_label = *label_input_matrix;
                float input[image_size][1];

                for (int i = 0; i < image_size; i++)
                {
                    input[i][0] = (float)((*(image_input_matrix + i)) / 255.0f);
                }
                float *ptr_input = &input[0][0];
                int guess = forward_propogation(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);

                weights_update(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, input_label);
                if (current_images % 600 == 0)
                {
                    printf("Predicted: %d | Actual: %d\n", guess, input_label);
                }
                if (guess == input_label)
                {
                    correct++;
                }
                free(image_input_matrix);
                free(label_input_matrix);
                free(ptr_output);

                if ((current_images + 1) % 600 == 0)
                {
                    printf("Epoch %d | Images Processed: %d | Accuracy: %.2f%%\n", e + 1, current_images + 1, ((float)correct / (current_images + 1)) * 100.0);
                }
            }
            printf("Epoch %d accuracy: %f\n", e + 1, ((float)correct / train_images) * 100.0);
        }
 //test for newly updated one weights and bias 
        int correct = 0;
        int confusion_matrix[10][10] = {0};
        int *confusion_matrix_ptr = &confusion_matrix[0][0];
        for (int current_images = 0; current_images < test_images; current_images++)
        {
            unsigned char *image_input_matrix = (unsigned char *)calloc(784, sizeof(unsigned char));
            unsigned char *label_input_matrix = (unsigned char *)calloc(1, sizeof(unsigned char));

            mnist_image_loader(file2, 784, image_input_matrix, current_images);
            mnist_label_loader(file2_label, 1, label_input_matrix, current_images);

            float *ptr_output = (float *)calloc(10, sizeof(float));

            int input_label = *label_input_matrix;
            float input[image_size][1];

            for (int i = 0; i < image_size; i++)
            {
                input[i][0] = (float)((*(image_input_matrix + i)) / 255.0f);
            }
            float *ptr_input = &input[0][0];
            int guess = forward_propogation(ptr_input, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);
            confusion_matrix[guess][input_label]++;
            if (guess == input_label)
            {
                correct++;
            }
            free(image_input_matrix);
            free(label_input_matrix);
            free(ptr_output);
        }
        printf("Testing accuracy: %f\n", ((float)correct / test_images) * 100.0);
        print_confusion_matrix(confusion_matrix_ptr);

        save_weight_bias("B24CS1078_B24CS1025_B24MT1042_B24CM1004_B24EE1053_weight_bias.bin", ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3);
        printf("Model updated successfully!!\n");
        break;
    }
    
    case 4:{
        float *input_image=calloc(784,sizeof(float));
        printf("Enter the image name with extension:-");
        char image_name[100];
        scanf("%s",image_name);
        input_image=user_given_image(image_name);

        load_weights_bias("B24CS1078_B24CS1025_B24MT1042_B24CM1004_B24EE1053_weight_bias.bin", ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3);
            float *ptr_output = (float *)calloc(10, sizeof(float));
            int guess = forward_propogation(input_image, ptr_w1, ptr_b1, ptr_w2, ptr_b2, ptr_w3, ptr_b3, ptr_output);
            printf("Predicted: %d\n", guess);
            free(ptr_output);
        break;
    }
    
    default:
        printf("Invalid choice!!\n\n");
        main();
        break;//in valid move
    }   
    //  thank you
    return 0;
}
